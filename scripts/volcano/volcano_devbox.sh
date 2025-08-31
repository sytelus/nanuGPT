#! /bin/bash

####################################################################################################
# This script submits a job to volcano cluster
# It copies the source directory to a shared location, substitutes placeholders in yaml config and
# submits the job to the cluster.

# Invoke this script as:
# ./scripts/volcano/volcano_submit.sh my_train.py some_args
####################################################################################################


set -eu -o pipefail -o xtrace # fail if any command failes, log all commands, -o xtrace

export JOB_NAME=${JOB_NAME:-devbox}
export NODES=${NODES:-1}
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
export CONTAINER_IMAGE_PATH=${CONTAINER_IMAGE_PATH:-"nvcr.io/nvidia/pytorch:25.08-py3"} #docker://@nvcr.io#nvidia/pytorch:24.07-py3
export VOLCANO_NAMESPACE=${VOLCANO_NAMESPACE:-} # namespace in volcano cluster
export VOLCANO_DATA_PVC_NAME=${VOLCANO_DATA_PVC_NAME:-} # data PVC claim in volcano cluster
export CONTAINER_PORT=${CONTAINER_PORT:-23456} # pytorch MASTER_PORT

export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

export MEMORY_SIZE_LIMIT=${MEMORY_SIZE_LIMIT:-100Gi}
export CPU_REQUESTS=${CPU_REQUESTS:-192}
export MEMORY_REQUESTS=${MEMORY_REQUESTS:-2600Gi}
export RDMA_REQUESTS=${RDMA_REQUESTS:-1}


if kubectl get vcjob "${JOB_NAME}" -n "${VOLCANO_NAMESPACE}" >/dev/null 2>&1; then
  echo "Job ${JOB_NAME} already exists in ${VOLCANO_NAMESPACE}"
  exit 1
fi

# directory where this script is running
SCRIPT_DIR="$(dirname "$(realpath "$0")")"

# create a temp working directory for rendered artifacts
TMP_DIR="$(mktemp -d -t volcano_devbox.XXXXXXXXXX)"
cleanup() { rm -rf "${TMP_DIR}"; }
trap cleanup EXIT

# number os workers = nodes - 1 (master node)
export WORKERS=$(( NODES - 1 ))

# output core variables so user can see what is being used
echo "CONTAINER_IMAGE_PATH: ${CONTAINER_IMAGE_PATH:-<not set>}"
echo "ENV_SETUP_SCRIPT: ${ENV_SETUP_SCRIPT:-<not set>}"
echo "NODES: ${NODES:-<not set>}"
echo "GPUS_PER_NODE: ${GPUS_PER_NODE:-<not set>}"

envsubst < "${SCRIPT_DIR}/volcano_job.yaml" | tee "${TMP_DIR}/volcano_rendered.yaml"

VCJOB_NAME=$(kubectl create -f "${TMP_DIR}/volcano_rendered.yaml" -o jsonpath='{.metadata.name}{"\n"}')
echo "Created VCJob: $VCJOB_NAME"

# drop into the node
kubectl exec -it $VCJOB_NAME -- bash

wait

