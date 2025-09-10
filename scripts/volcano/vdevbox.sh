#! /bin/bash

####################################################################################################
# This script submits a job to volcano cluster
# It copies the source directory to a shared location, substitutes placeholders in yaml config and
# submits the job to the cluster.

# Invoke this script as:
# ./scripts/volcano/volcano_submit.sh my_train.py some_args
####################################################################################################


set -eu -o pipefail # -o xtrace # fail if any command failes, log all commands, -o xtrace

export USER_ALIAS=${USER%@*}
export NODES=${NODES:-1}
export VOLCANO_NAMESPACE=${VOLCANO_NAMESPACE:-} # namespace in volcano cluster
export VOLCANO_DATA_PVC_NAME=${VOLCANO_DATA_PVC_NAME:-} # data PVC claim in volcano cluster
export WANDB_API_KEY=${WANDB_API_KEY:-}
export WANDB_HOST=${WANDB_HOST:-}
export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}

export CONTAINER_PORT=${CONTAINER_PORT:-23456} # pytorch MASTER_PORT
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

for arg in "$@"; do
  if [[ "$arg" == "--cpu" ]]; then
    # CPU only devbox
    export JOB_NAME=${USER_ALIAS}-${JOB_NAME:-devbox-cpu}
    export GPUS_PER_NODE=${GPUS_PER_NODE:-0}
    export CONTAINER_IMAGE_PATH=${CONTAINER_IMAGE_PATH:-"joelewhite/az-cli-ubuntu:latest"} #docker://@nvcr.io#nvidia/pytorch:24.07-py3

    export MEMORY_SIZE_LIMIT=${MEMORY_SIZE_LIMIT:-8Gi}
    export CPU_REQUESTS=${CPU_REQUESTS:-12}
    export MEMORY_REQUESTS=${MEMORY_REQUESTS:-64Gi}
    export RDMA_REQUESTS=${RDMA_REQUESTS:-0}
  else
    export JOB_NAME=${USER_ALIAS}-${JOB_NAME:-devbox}
    export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
    export CONTAINER_IMAGE_PATH=${CONTAINER_IMAGE_PATH:-"nvcr.io/nvidia/pytorch:25.08-py3"} #docker://@nvcr.io#nvidia/pytorch:24.07-py3

    export MEMORY_SIZE_LIMIT=${MEMORY_SIZE_LIMIT:-100Gi}
    export CPU_REQUESTS=${CPU_REQUESTS:-192}
    export MEMORY_REQUESTS=${MEMORY_REQUESTS:-2600Gi}
    export RDMA_REQUESTS=${RDMA_REQUESTS:-1}
  fi
done

if kubectl get vcjob "${JOB_NAME}" -n "${VOLCANO_NAMESPACE}" >/dev/null 2>&1; then
  echo "Job ${JOB_NAME} already exists in ${VOLCANO_NAMESPACE}"
  exit 1
fi

# directory where this script is running
SCRIPT_DIR="$(dirname "$(realpath "$0")")"

# create a temp working directory for rendered artifacts
# Cleanup handler
VCJOB_FQN=""
TMP_DIR="$(mktemp -d -t volcano_devbox.XXXXXXXXXX)"
cleanup() {
  local rc=$?
  set +e
  rm -rf "${TMP_DIR}";
  if [[ -n "${VCJOB_FQN}" ]]; then
    echo "[cleanup] Deleting ${VCJOB_FQN} ..."
    kubectl -n "${VOLCANO_NAMESPACE}" delete "${VCJOB_FQN}" --ignore-not-found=true >/dev/null 2>&1 || true
    # Best-effort wait for pods to vanish
    kubectl -n "${VOLCANO_NAMESPACE}" wait --for=delete pod -l volcano.sh/job-name="${JOB_NAME}" --timeout=120s >/dev/null 2>&1 || true
  fi
  exit ${rc}
}
trap cleanup EXIT

# number os workers = nodes - 1 (master node)
export WORKERS=$(( NODES - 1 ))

export ENV_VARS=""
make_env_vars() {
    local env_vars_val="export"
    local var val
    for var in "$@"; do
        # Check if variable is set
        if [ -v "$var" ]; then
            # Safely expand its value
            val=${!var}
            # Append to ENV_VARS
            env_vars_val+=" $var=\"${val}\""
        fi
    done
    if [[ "${env_vars_val}" == "export" ]]; then
      env_vars_val=""
    fi
    # Export ENV_VARS itself
    export ENV_VARS=${env_vars_val}
}
# make ENV_VARS variable that will be script to setup env in container
make_env_vars CUDA_LAUNCH_BLOCKING TORCHINDUCTOR_COORDINATE_DESCENT_TUNING TORCHINDUCTOR_COORDINATE_DESCENT_CHECK_ALL_DIRECTIONS TORCHINDUCTOR_COORDINATE_DESCENT_RADIUS \
    WANDB_API_KEY WANDB_HOST
echo "ENV_VARS to be setup in container:"
echo "--------------------------------"
echo "$ENV_VARS"
echo "--------------------------------"

# output core variables so user can see what is being used
echo "CONTAINER_IMAGE_PATH: ${CONTAINER_IMAGE_PATH:-<not set>}"
echo "ENV_SETUP_SCRIPT: ${ENV_SETUP_SCRIPT:-<not set>}"
echo "NODES: ${NODES:-<not set>}"
echo "GPUS_PER_NODE: ${GPUS_PER_NODE:-<not set>}"

envsubst < "${SCRIPT_DIR}/volcano_devbox.yaml" > "${TMP_DIR}/volcano_rendered.yaml"

VCJOB_FQN=$(kubectl create -f "${TMP_DIR}/volcano_rendered.yaml" -o name)
echo "Created: $VCJOB_FQN"

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

set +e; kubectl -n "${VOLCANO_NAMESPACE}" exec -it "${POD_NAME}" -c master -- bash; rc=$?; set -e; echo "session exit code rc=$rc"

wait

