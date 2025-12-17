#!/usr/bin/env bash


# Filename: copy2pvc.sh
# USAGE: copy2pvc /path/on/jumpbox [remote/path/in/pvc]

set -euo pipefail

### --- Config via environment ---
: "${VOLCANO_DATA_PVC_NAME:?Set VOLCANO_DATA_PVC_NAME to your PVC claim name}"
: "${VOLCANO_NAMESPACE:?Set VOLCANO_NAMESPACE to your target namespace}"

# Resource defaults (aligned with volcano_job.yaml usage)
GPUS_PER_NODE=0
CPU_REQUESTS=12 # typical default 192
MEMORY_REQUESTS=64Gi # typical default 2600Gi
RDMA_REQUESTS=0
MEMORY_SIZE_LIMIT=8Gi # typical default 100Gi
CONTAINER_IMAGE=busybox:1.36

### --- Args ---
LOCAL_PATH="${1:-}"
ABS_LOCAL="$(readlink -f "${LOCAL_PATH}")"
TARGET_BASENAME="$(basename "${ABS_LOCAL}")"
REMOTE_PATH="${2:-${TARGET_BASENAME}}" # should be of the form "abc/xyz/pqr" after which TARGET_BASENAME is enforced


if [[ -z "${LOCAL_PATH}" ]]; then
  echo "Usage: $0 <local-path>" >&2
  echo "Requires env: VOLCANO_DATA_PVC_NAME, VOLCANO_NAMESPACE" >&2
  exit 1
fi
if [[ ! -d "${LOCAL_PATH}" ]]; then
  echo "Local path '${LOCAL_PATH}' does not exist or is not a directory." >&2
  exit 1
fi

# Compute target dir inside PVC

PVC_MOUNT="/mnt/pvc"
PVC_TARGET_DIR="${PVC_MOUNT%/}/${REMOTE_PATH}"

export USER_ALIAS=${USER%@*}
JOB_NAME=${USER_ALIAS}-${JOB_NAME:-pvc-loader}

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
echo "Creating Volcano Job for PVC copy..."
VCJOB_FQN="$(
kubectl create -f - -n "${VOLCANO_NAMESPACE}" -o name <<YAML
apiVersion: batch.volcano.sh/v1alpha1
kind: Job
metadata:
  generateName: ${JOB_NAME}
  namespace: ${VOLCANO_NAMESPACE}
  labels:
    submitter: "${USER_ALIAS}"
spec:
  queue: ${VOLCANO_NAMESPACE}
  minAvailable: 1
  plugins:
    ssh: []        # passwordless SSH + /etc/volcano hostfiles
    svc: []        # headless Services when containerPorts exist
    env: []        # VC_* envs (host lists, etc.)
  tasks:
    - name: loader
      replicas: 1
      template:
        metadata:
          labels:
            app: ${JOB_NAME}
            role: master
        spec:
          schedulerName: volcano
          restartPolicy: Never
          volumes:
            - name: dshm
              emptyDir:
                medium: Memory
                sizeLimit: ${MEMORY_SIZE_LIMIT}
            - name: data
              persistentVolumeClaim:
                claimName: ${VOLCANO_DATA_PVC_NAME}
          # tolerations:
          #   - key: "rdma"
          #     operator: "Exists"
          #     effect: "NoSchedule"
          #   - key: "nvidia.com/gpu"
          #     operator: "Exists"
          #     effect: "NoSchedule"
          affinity:
            podAntiAffinity:
              requiredDuringSchedulingIgnoredDuringExecution:
                - labelSelector:
                    matchLabels:
                      app: ${JOB_NAME}
                  topologyKey: kubernetes.io/hostname
          containers:
            - name: loader
              image: ${CONTAINER_IMAGE}
              imagePullPolicy: IfNotPresent
              command: ["/bin/sh","-lc"]
              # workingDir: ${PVC_MOUNT}
              args:
                - |
                  set -eu -o pipefail -o xtrace # fail if any command failes, log all commands, -o xtrace
                  echo "[\$(date -u +%FT%TZ)] ${JOB_NAME} starting; PVC at ${PVC_MOUNT}"
                  # mkdir -p ${PVC_TARGET_DIR} # no longer needed as we do this inline before copy files
                  echo "[\$(date -u +%FT%TZ)] target dir ready: ${PVC_MOUNT}/${PVC_TARGET_DIR}"
                  echo "[\$(date -u +%FT%TZ)] going to sleep (infinite) so you can upload"
                  sleep infinity
              volumeMounts:
                - name: dshm
                  mountPath: /dev/shm
                - name: data
                  mountPath: ${PVC_MOUNT}
              resources:
                requests: &requests
                  # nvidia.com/gpu: "${GPUS_PER_NODE}"
                  # rdma/rdma_shared_device_a: "${RDMA_REQUESTS}"
                  cpu: "${CPU_REQUESTS}"
                  memory: "${MEMORY_REQUESTS}"
                limits: *requests
YAML
)"
echo "Created: ${VCJOB_FQN}"

# Extract job name (vcjob/<name> -> <name>) for selectors
VCJOB_NAME="${VCJOB_FQN#*/}"
echo "Volcano Job name: ${VCJOB_NAME}"

# Track pod creation immediately (busy cluster aware)
echo "Waiting for loader pod to be scheduled..."
POD_NAME=""
while [[ -z "${POD_NAME}" ]]; do
  POD_POD_NAME="$(kubectl -n "${VOLCANO_NAMESPACE}" get pods -l "volcano.sh/job-name=${VCJOB_NAME}" -o name 2>/dev/null || true)"
  POD_NAME="${POD_POD_NAME#*/}"
  sleep 10
done
echo "Pod: ${POD_NAME}"

# Wait until the pod is Ready
echo "Waiting for pod to be Ready..."
kubectl -n "${VOLCANO_NAMESPACE}" wait --for=condition=Ready "${POD_POD_NAME}" --timeout=10m

echo
echo "=== Loader pod initial logs ==="
kubectl -n "${VOLCANO_NAMESPACE}" logs "${POD_NAME}" || true
echo "=== End logs ==="
echo

# Stream copy with mkdir + tar in a single exec (race-free)
echo "Uploading '${ABS_LOCAL}' -> '${PVC_TARGET_DIR}' ..."
tar -C "${ABS_LOCAL}" -cf - . \
| kubectl -n "${VOLCANO_NAMESPACE}" exec -i "${POD_NAME}" -- sh -lc "mkdir -p '${PVC_TARGET_DIR}' && tar -C '${PVC_TARGET_DIR}' -xpf -"

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