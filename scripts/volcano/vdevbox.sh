#! /bin/bash
set -eu -o pipefail # -o xtrace # fail if any command failes, log all commands, -o xtrace

####################################################################################################
# This script takes node(s) from cluster and gives user an interactive bash shell.
#
# Invoke this script as:
# ----------------------
# vdevbox.sh # gets 1 node GPU devbox
# vdevbox.sh --cpu # gets 1 node CPU only devbox
#
# Useful variables:
# -----------------
# NODES - number of nodes to use, default 1
# GPUS_PER_NODE - number of gpus per node, default 8
# CONTAINER_IMAGE_PATH - container image to use, default nvcr.io/nvidia/pytorch:25.08-py3
# VOLCANO_NAMESPACE - namespace in volcano cluster
# VOLCANO_DATA_PVC_NAME - data PVC claim in volcano cluster
####################################################################################################

export USER_ALIAS=${USER%@*}
export NODES=${NODES:-1}
export VOLCANO_NAMESPACE=${VOLCANO_NAMESPACE:-} # namespace in volcano cluster
export VOLCANO_DATA_PVC_NAME=${VOLCANO_DATA_PVC_NAME:-} # data PVC claim in volcano cluster
export TRANSFER_VARS=${TRANSFER_VARS:-} # space separated list of additional env vars to transfer to container

export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}
export CONTAINER_PORT=${CONTAINER_PORT:-23456} # pytorch MASTER_PORT
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

if [ "$#" -eq 0 ]; then
  # prompt user that GPU devboxes are expensive and should not be kept alivve beyond short duration and
  # if they agree with responsible usage then proceed else ask them to use --cpu switch and exit
  echo "You are about to start a GPU devbox. This is expensive and MUST not be kept alive beyond short duration."
  read -p "Are you aware about responsible use guidelines for this cluster? (y/n): " -n 1 -r
  echo    # (optional) move to a new line
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
      echo "Please use --cpu switch to start a CPU only devbox. Contact Shital or Sahaj for learning more about responsible usage."
      exit 1
  fi

  # default GPU devbox vars
  export JOB_NAME=${USER_ALIAS}-devbox
  export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
  export CONTAINER_IMAGE_PATH=${CONTAINER_IMAGE_PATH:-"nvcr.io/nvidia/pytorch:25.08-py3"} #docker://@nvcr.io#nvidia/pytorch:24.07-py3

  export MEMORY_SIZE_LIMIT=${MEMORY_SIZE_LIMIT:-100Gi}
  export CPU_REQUESTS=${CPU_REQUESTS:-192}
  export MEMORY_REQUESTS=${MEMORY_REQUESTS:-2600Gi}
  export RDMA_REQUESTS=${RDMA_REQUESTS:-1}

  # good defaults for Pytorch
  # avoid OOM errors by allowing segments to expand
  export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
  # turn on heavy optimizations in torchinductor
  # export TORCHINDUCTOR_COORDINATE_DESCENT_TUNING=${TORCHINDUCTOR_COORDINATE_DESCENT_TUNING:-1}

  export TOLERENCE_YAML='tolerations:
            - key: "nvidia.com/gpu"
              operator: "Exists"
              effect: "NoSchedule"
'
  # if RDMA_REQUESTS is 1 then add nvidia gpu toleration as well
  if [[ "${RDMA_REQUESTS}" -eq 1 ]]; then
    TOLERENCE_YAML+='
            - key: "rdma"
              operator: "Exists"
              effect: "NoSchedule"
'
  fi
  export RDMA_YAML="rdma/rdma_shared_device_a: \"${RDMA_REQUESTS}\""
else
  if [[ "$1" == "--cpu" ]]; then
    # CPU only devbox
    export JOB_NAME=${USER_ALIAS}-devbox-cpu
    export GPUS_PER_NODE=${GPUS_PER_NODE:-0}
    export CONTAINER_IMAGE_PATH=${CONTAINER_IMAGE_PATH:-"sytelus/cpu-devbox:2025.09.26"} #docker://@nvcr.io#nvidia/pytorch:24.07-py3

    export MEMORY_SIZE_LIMIT=${MEMORY_SIZE_LIMIT:-8Gi}
    export CPU_REQUESTS=${CPU_REQUESTS:-12}
    export MEMORY_REQUESTS=${MEMORY_REQUESTS:-64Gi}
    export RDMA_REQUESTS=${RDMA_REQUESTS:-0}

    export TOLERENCE_YAML=''
    export RDMA_YAML=''
  else
    echo "Usage: $0 [--cpu]" >&2
    exit 1
  fi
fi

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
make_env_vars ${TRANSFER_VARS} CUDA_LAUNCH_BLOCKING TORCHINDUCTOR_COORDINATE_DESCENT_TUNING \
  TORCHINDUCTOR_COORDINATE_DESCENT_CHECK_ALL_DIRECTIONS TORCHINDUCTOR_COORDINATE_DESCENT_RADIUS \
  PYTORCH_CUDA_ALLOC_CONF TORCHINDUCTOR_AUTOTUNE_IN_SUBPROC TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS

echo "ENV_VARS to be setup in container:"
echo "--------------------------------"
echo "$ENV_VARS"
echo "--------------------------------"

echo "Using:"
echo "  JOB_NAME: ${JOB_NAME}"
echo "  NODES: ${NODES}"
echo "  GPUS_PER_NODE: ${GPUS_PER_NODE}"
echo "  CONTAINER_IMAGE_PATH: ${CONTAINER_IMAGE_PATH}"
echo "  VOLCANO_NAMESPACE: ${VOLCANO_NAMESPACE:-<not set>}"
echo "  VOLCANO_DATA_PVC_NAME: ${VOLCANO_DATA_PVC_NAME:-<not set>}"
echo "  CPU_REQUESTS: ${CPU_REQUESTS}"
echo "  MEMORY_REQUESTS: ${MEMORY_REQUESTS}"
echo "  RDMA_REQUESTS: ${RDMA_REQUESTS}"
echo "  MEMORY_SIZE_LIMIT: ${MEMORY_SIZE_LIMIT}"

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
if ! kubectl -n "${VOLCANO_NAMESPACE}" wait --for=condition=Ready "${POD_POD_NAME}" --timeout=100m; then
  if ! kubectl -n "${VOLCANO_NAMESPACE}" get pod "${POD_NAME}" -o jsonpath='{.status.containerStatuses[0].state.running}' 2>/dev/null | grep -q 'true'; then
    echo "Pod ${POD_NAME} is not running. Dumping pod status:"
    kubectl -n "${VOLCANO_NAMESPACE}" describe pod "${POD_NAME}" || true
    echo "Dumping pod events:"
    kubectl -n "${VOLCANO_NAMESPACE}" get events --sort-by=.lastTimestamp --field-selector involvedObject.name="${POD_NAME}" || true
  fi
  exit 1
fi

echo
echo "=== Loader pod initial logs ==="
kubectl -n "${VOLCANO_NAMESPACE}" logs "${POD_NAME}" || true
echo "=== End logs ==="
echo

set +e; kubectl -n "${VOLCANO_NAMESPACE}" exec -it "${POD_NAME}" -c master -- bash; rc=$?; set -e; echo "session exit code rc=$rc"

wait

