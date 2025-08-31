#!/usr/bin/env bash
set -euo pipefail

### --- Config via environment ---
: "${VOLCANO_DATA_PVC_NAME:?Set VOLCANO_DATA_PVC_NAME to your PVC claim name}"
: "${VOLCANO_NAMESPACE:=default}"

# Resource defaults
: "${GPUS_PER_NODE:=0}"             # >0 only if GPUs are actually available
: "${CPU_REQUESTS:=500m}"
: "${MEMORY_REQUESTS:=2Gi}"
: "${RDMA_REQUESTS:=0}"             # >0 only if RDMA device plugin is present

### --- Args ---
LOCAL_PATH="${1:-}"

if [[ -z "${LOCAL_PATH}" ]]; then
  echo "Usage: $0 <local-path>" >&2
  exit 1
fi
if [[ ! -d "${LOCAL_PATH}" ]]; then
  echo "Local path '${LOCAL_PATH}' does not exist or is not a directory." >&2
  exit 1
fi

# Compute target dir inside PVC
ABS_LOCAL="$(readlink -f "${LOCAL_PATH}")"
PVC_TARGET_BASENAME="$(basename "${ABS_LOCAL}")"
PVC_TARGET_DIR="/mnt/pvc/${PVC_TARGET_BASENAME}"

POD_SLEEP_CMD='while true; do sleep 3600; done'

STAMP="$(date +%Y%m%d-%H%M%S)"
JOB_NAME="pvc-loader-${STAMP}"

echo "Namespace:            ${VOLCANO_NAMESPACE}"
echo "PVC claim:            ${VOLCANO_DATA_PVC_NAME}"
echo "Volcano Job:          ${JOB_NAME}"
echo "Local path:           ${ABS_LOCAL}"
echo "PVC target directory: ${PVC_TARGET_DIR}"
echo "Pod sleep:            infinite"
echo "Resources:            GPU=${GPUS_PER_NODE} CPU=${CPU_REQUESTS} MEM=${MEMORY_REQUESTS} RDMA=${RDMA_REQUESTS}"
echo

# Cleanup handler
VCJOB_FQN=""
cleanup() {
  local rc=$?
  set +e
  if [[ -n "${VCJOB_FQN}" ]]; then
    echo "[cleanup] Deleting ${VCJOB_FQN} ..."
    kubectl -n "${VOLCANO_NAMESPACE}" delete "${VCJOB_FQN}" --ignore-not-found=true >/dev/null 2>&1 || true
    # Best-effort wait for pods to vanish
    kubectl -n "${VOLCANO_NAMESPACE}" wait --for=delete pod -l volcano.sh/job-name="${JOB_NAME}" --timeout=120s >/dev/null 2>&1 || true
  fi
  exit $rc
}
trap cleanup EXIT

# Create Volcano Job via stdin, capture name with -o name
echo "Creating Volcano Job..."
VCJOB_FQN="$(
kubectl create -f - -n "${VOLCANO_NAMESPACE}" -o name <<YAML
apiVersion: batch.volcano.sh/v1alpha1
kind: Job
metadata:
  name: ${JOB_NAME}
spec:
  schedulerName: volcano
  minAvailable: 1
  tasks:
    - name: loader
      replicas: 1
      template:
        metadata:
          labels:
            app: pvc-loader
        spec:
          restartPolicy: Never
          volumes:
            - name: data
              persistentVolumeClaim:
                claimName: ${VOLCANO_DATA_PVC_NAME}
          containers:
            - name: loader
              image: busybox:1.36
              command: ["/bin/sh","-lc"]
              args:
                - |
                  set -euo pipefail
                  echo "[\$(date -u +%FT%TZ)] pvc-loader starting; PVC at /mnt/pvc"
                  mkdir -p ${PVC_TARGET_DIR}
                  echo "[\$(date -u +%FT%TZ)] target dir ready: ${PVC_TARGET_DIR}"
                  echo "[\$(date -u +%FT%TZ)] going to sleep (infinite) so you can upload"
                  ${POD_SLEEP_CMD}
              volumeMounts:
                - name: data
                  mountPath: /mnt/pvc
              resources:
                requests: &requests
                  nvidia.com/gpu: "${GPUS_PER_NODE}"
                  cpu: "${CPU_REQUESTS}"
                  memory: "${MEMORY_REQUESTS}"
                  rdma/rdma_shared_device_a: "${RDMA_REQUESTS}"
                limits: *requests
YAML
)"
echo "Created: ${VCJOB_FQN}"

# Extract job name (vcjob/<name> -> <name>) for selectors
VCJOB_NAME="${VCJOB_FQN#*/}"

# Track pod creation immediately (busy cluster aware)
echo "Waiting for loader pod to be scheduled..."
POD_NAME=""
DEADLINE=$((SECONDS + 600))  # 10 minutes max
while [[ -z "${POD_NAME}" ]]; do
  POD_NAME="$(kubectl -n "${VOLCANO_NAMESPACE}" get pods -l volcano.sh/job-name="${JOB_NAME}" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
  if [[ ${SECONDS} -ge ${DEADLINE} ]]; then
    echo "Timed out waiting for pod creation for job ${JOB_NAME}" >&2
    exit 1
  fi
  sleep 1
done
echo "Pod: ${POD_NAME}"

# Wait until the pod is Ready
echo "Waiting for pod to be Ready..."
kubectl -n "${VOLCANO_NAMESPACE}" wait --for=condition=Ready "pod/${POD_NAME}" --timeout=10m

echo
echo "=== Loader pod initial logs ==="
kubectl -n "${VOLCANO_NAMESPACE}" logs "${POD_NAME}" || true
echo "=== End logs ==="
echo

# Stream copy with tar -> tar; errors cause immediate exit due to pipefail
echo "Uploading '${ABS_LOCAL}' -> '${PVC_TARGET_DIR}' ..."
tar -C "${ABS_LOCAL}" -cf - . \
| kubectl -n "${VOLCANO_NAMESPACE}" exec -i "${POD_NAME}" -- tar -C "${PVC_TARGET_DIR}" -xpf -

echo "Upload complete. Verifying listing:"
kubectl -n "${VOLCANO_NAMESPACE}" exec "${POD_NAME}" -- /bin/sh -lc "ls -lah ${PVC_TARGET_DIR} | head -50" || true

# Show tail of logs (optional)
echo
echo "=== Loader pod logs (tail) ==="
kubectl -n "${VOLCANO_NAMESPACE}" logs --tail=50 "${POD_NAME}" || true
echo "=== End logs ==="

echo
echo "Success. Cleaning up the helper job/pod..."
# cleanup trap will delete the vcjob and wait for pods to vanish
