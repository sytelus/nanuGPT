#! /bin/bash

####################################################################################################
# This script submits a job to volcano cluster
# It copies the source directory to a shared location, substitutes placeholders in yaml config and
# submits the job to the cluster.

# Invoke this script as:
# ./scripts/volcano/volcano_submit.sh my_train.py some_args
####################################################################################################


set -eu -o pipefail # -o xtrace # fail if any command failes, log all commands, -o xtrace

REQUIRED_VARS=("OUT_DIR") # out_dir specified where source code will be copied and job outputs will be stored, sypically mount shared to the cluster
SOURCE_DIR=${SOURCE_DIR:-.} # where is source directory
USER_ALIAS=${USER%@*}
export JOB_NAME=${JOB_NAME:-${USER_ALIAS}-ok-to-kill-test-job}
export START_COMMAND=${START_COMMAND:-"$@"} # use all args to this script as command we will execute
export DATA_ROOT=${DATA_ROOT:-} # data directory to mount in container
export NODES=${NODES:-1}
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
export CONTAINER_IMAGE_PATH=${CONTAINER_IMAGE_PATH:-"nvcr.io/nvidia/pytorch:25.06-py3"} #docker://@nvcr.io#nvidia/pytorch:24.07-py3
export ENV_SETUP_SCRIPT=${ENV_SETUP_SCRIPT:-} # script to setup environment for specific cluster, this runs before any code
export VOLCANO_NAMESPACE=${VOLCANO_NAMESPACE:-} # namespace in volcano cluster
export VOLCANO_DATA_PVC_NAME=${VOLCANO_DATA_PVC_NAME:-} # data PVC claim in volcano cluster
export WANDB_API_KEY=${WANDB_API_KEY:-}
export WANDB_HOST=${WANDB_HOST:-}

export INSTALL_PACKAGE=${INSTALL_PACKAGE:-1} # assume source directory is package and first do pip install -e .
export UPDATE_PYTHONPATH=${UPDATE_PYTHONPATH:-0} # add source dir to PYTHONPATH (ignored if INSTALL_PACKAGE=1)

export CONTAINER_PORT=${CONTAINER_PORT:-23456} # pytorch MASTER_PORT
export NPROC_PER_NODE=${NPROC_PER_NODE:-8}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

export MEMORY_SIZE_LIMIT=${MEMORY_SIZE_LIMIT:-100Gi}
export CPU_REQUESTS=${CPU_REQUESTS:-192}
export MEMORY_REQUESTS=${MEMORY_REQUESTS:-2600Gi}
export RDMA_REQUESTS=${RDMA_REQUESTS:-1}

### ---------- Check required environment variables
for var in "${REQUIRED_VARS[@]}"; do
    [ -z "${!var}" ] && { echo "Error: Required environment variable '$var' is not set." >&2; exit 1; }
done
### ---------- End check required environment variables

if kubectl get vcjob "${JOB_NAME}" -n "${VOLCANO_NAMESPACE}" >/dev/null 2>&1; then
  echo "Job ${JOB_NAME} already exists in ${VOLCANO_NAMESPACE}"
  exit 1
fi

# directory where this script is running
SCRIPT_DIR="$(dirname "$(realpath "$0")")"

export USER_ALIAS=${USER%@*}
JOB_NAME=${USER_ALIAS}-${JOB_NAME:-test-job}

# number os workers = nodes - 1 (master node)
export WORKERS=$(( NODES - 1 ))

# create sub dir for this specific run in our dir
export JOB_OUT_DIR=runs/${USER_ALIAS}/${JOB_NAME}-$(date +%Y-%m-%d_%H-%M-%S_%3N)
LOCAL_JOB_OUT_DIR="${OUT_DIR}/${JOB_OUT_DIR}"
rm -rf "${LOCAL_JOB_OUT_DIR}"
mkdir -p "${LOCAL_JOB_OUT_DIR}"

# output core variables so user can see what is being used
echo "SOURCE_DIR: $(realpath ${SOURCE_DIR:-<not set>})"
echo "START_COMMAND: ${START_COMMAND:-<not set>}"
echo "JOB_OUT_DIR: ${JOB_OUT_DIR:-<not set>}"
echo "LOCAL_JOB_OUT_DIR: ${LOCAL_JOB_OUT_DIR:-<not set>}"
echo "DATA_ROOT: ${DATA_ROOT:-<not set>}"
echo "CONTAINER_IMAGE_PATH: ${CONTAINER_IMAGE_PATH:-<not set>}"
echo "ENV_SETUP_SCRIPT: ${ENV_SETUP_SCRIPT:-<not set>}"
echo "NODES: ${NODES:-<not set>}"
echo "GPUS_PER_NODE: ${GPUS_PER_NODE:-<not set>}"

# some clusters may need additional env vars in which case specify script that sets them
if [ ! -z "${ENV_SETUP_SCRIPT}" ]; then
    # copy to job out dir so its available in the container
    cp "${ENV_SETUP_SCRIPT}" "${LOCAL_JOB_OUT_DIR}/env_setup.sh"
fi

# We wll also install package requirements from this directory
TARGET_SOURCE_DIR="${LOCAL_JOB_OUT_DIR}/source_dir"
# remove existing directory
rm -rf "${TARGET_SOURCE_DIR}"
mkdir -p "${TARGET_SOURCE_DIR}"
pushd "${SOURCE_DIR}"
# if all commited changes then get latest head and create archive, else copy except .git folder
# if git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
#     echo "Copying source from git repo using archive..."
#     git archive --format=tar HEAD | tar -x -C "${TARGET_SOURCE_DIR}"
# else
    echo "Copying source from working dir..."
    rsync -a --exclude='.git' ./ "${TARGET_SOURCE_DIR}"
#fi
popd

# copy volcano scripts to job out dir so we can use it in cluster
VOLCANO_SCRIPT_DIR="${LOCAL_JOB_OUT_DIR}/volcano_scripts"
# copy all slurm scripts to job out dir so we can use them in cluster
rm -rf "${VOLCANO_SCRIPT_DIR}"
mkdir -p "${VOLCANO_SCRIPT_DIR}"
cp "${SCRIPT_DIR}/"*.sh "${VOLCANO_SCRIPT_DIR}/"
chmod +x "${VOLCANO_SCRIPT_DIR}/"*.sh

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

envsubst < "${SCRIPT_DIR}/volcano_job.yaml" > "${LOCAL_JOB_OUT_DIR}/volcano_rendered.yaml"

# now that direcoty is ready, we need to copy this to pvc available to cluster
echo "Copying source and scripts from ${LOCAL_JOB_OUT_DIR} to PVC ${VOLCANO_DATA_PVC_NAME}..."
JOB_NAME=pvc-copy-${JOB_NAME} bash ${SCRIPT_DIR}/copy2pvc.sh ${LOCAL_JOB_OUT_DIR} ${JOB_OUT_DIR}
echo "Copy to pvc complete."

VCJOB_FQN=$(kubectl create -f "${LOCAL_JOB_OUT_DIR}/volcano_rendered.yaml" -o name)
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

# Tail logs from a specific container (e.g., "trainer") in all pods of this job
for P in $(kubectl get pods -l volcano.sh/job-name="$VCJOB_NAME" -o name); do
  echo "=== $P ==="
  kubectl logs -f "$P" &
done
wait


