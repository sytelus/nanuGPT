#! /bin/bash
set -eu -o pipefail # -o xtrace # fail if any command failes, log all commands, -o xtrace

####################################################################################################
# This script submits a job to volcano cluster.
# It copies the source directory to cluster, installs the directory as editable package and
# runs the specified command.
#
# Invoke this script as:
# ----------------------
# JOB_NAME='my-project' NODES=2 \
# TRANSFER_VARS="DATA_ROOT WANDB_API_KEY WANDB_HOST" \
# vsubmit.sh my_train.py some_args
#
# Useful variables:
# -----------------
# JOB_NAME - must be set by user, used in job name
# OUT_DIR - output dir on cluster, default /data/<user>
# INSTALL_PACKAGE - if 1 (default) then pip install -e . the source dir before running command
# UPDATE_PYTHONPATH - if 1 then add source dir to PYTHONPATH (ignored if INSTALL_PACKAGE=1)
# TRANSFER_VARS - space separated list of names for env vars to transfer to container
#                 (e.g. DATA_ROOT WANDB_API_KEY WANDB_HOST)
#                 All these vars will be set on cluster to same value in current environment
# START_COMMAND - command to run in container, default all args to this script
# NODES - number of nodes to use, default 1
# GPUS_PER_NODE - number of gpus per node, default 8
# CONTAINER_IMAGE_PATH - container image to use, default nvcr.io/nvidia/pytorch:25.08-py3
# ENV_SETUP_SCRIPT - script to setup environment before running command
# VOLCANO_NAMESPACE - namespace in volcano cluster
# VOLCANO_DATA_PVC_NAME - data PVC claim in volcano cluster
####################################################################################################


SOURCE_DIR=${SOURCE_DIR:-.} # where is source directory
USER_ALIAS=${USER%@*}

OUT_DIR=${OUT_DIR:-/data/${USER_ALIAS}} # base output directory where we will create sub dir for this run
echo "Job output will be at '${OUT_DIR}' on cluster. If you don't want this then set OUT_DIR env var."

# Validate JOB_NAME
if [ -z "${JOB_NAME:-}" ]; then
  cat >&2 <<'EOF'

You must set JOB_NAME variable that will be used in your job names. You can do this by:

 JOB_NAME='my-project' vsubmit.sh <command_to_run_with_args>

NOTE: my-project must have alpha-numeric chars or - (no underscores or spaces)

EOF
  exit 1
fi

export JOB_NAME_FULL=${USER_ALIAS}-${JOB_NAME}
export TRANSFER_VARS=${TRANSFER_VARS:-} # space separated list of additional env vars to transfer to container
# Build START_COMMAND by shell-escaping each original arg, preserving spaces/quotes
if [[ -z "${START_COMMAND:-}" ]]; then
  START_COMMAND=""
  for arg in "$@"; do
    printf -v _q '%q' "$arg"
    START_COMMAND+=" ${_q}"
  done
  START_COMMAND="${START_COMMAND# }"
fi
export START_COMMAND # used later in the rendered YAML
export NODES=${NODES:-1}
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
export NPROC_PER_NODE=${NPROC_PER_NODE:-${GPUS_PER_NODE}} # by default use all gpus on node
export CONTAINER_IMAGE_PATH=${CONTAINER_IMAGE_PATH:-"nvcr.io/nvidia/pytorch:25.08-py3"} #docker://@nvcr.io#nvidia/pytorch:24.07-py3
export ENV_SETUP_SCRIPT=${ENV_SETUP_SCRIPT:-} # script to setup environment for specific cluster, this runs before any code
export VOLCANO_NAMESPACE=${VOLCANO_NAMESPACE:-} # namespace in volcano cluster
export VOLCANO_DATA_PVC_NAME=${VOLCANO_DATA_PVC_NAME:-} # data PVC claim in volcano cluster

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

# good defaults for Pytorch
# avoid OOM errors by allowing segments to expand
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
# turn on heavy optimizations in torchinductor
export TORCHINDUCTOR_COORDINATE_DESCENT_TUNING=${TORCHINDUCTOR_COORDINATE_DESCENT_TUNING:-1}

# validate START_COMMAND is not empty
if [ -z "${START_COMMAND}" ]; then
  echo "Error: You must specify command to run on cluster as argument to this script or set START_COMMAND env var.\nFor example: JOB_NAME='my-project' vsubmit.sh <command_to_run_with_args>"
  exit 1
fi

if kubectl get vcjob "${JOB_NAME_FULL}" -n "${VOLCANO_NAMESPACE}" >/dev/null 2>&1; then
  echo "Job ${JOB_NAME_FULL} already exists in ${VOLCANO_NAMESPACE}"
  exit 1
fi

# directory where this script is running
SCRIPT_DIR="$(dirname "$(realpath "$0")")"

# number os workers = nodes - 1 (master node)
export WORKERS=$(( NODES - 1 ))

# create sub dir for this specific run in our dir
export JOB_OUT_DIR=runs/${USER_ALIAS}/${JOB_NAME_FULL}-$(date +%Y-%m-%d_%H-%M-%S_%3N)
LOCAL_JOB_OUT_DIR="${OUT_DIR}/${JOB_OUT_DIR}"
rm -rf "${LOCAL_JOB_OUT_DIR}"
mkdir -p "${LOCAL_JOB_OUT_DIR}"

# output core variables so user can see what is being used
echo "SOURCE_DIR: $(realpath ${SOURCE_DIR:-<not set>})"
echo "START_COMMAND: ${START_COMMAND:-<not set>}"
echo "JOB_OUT_DIR: ${JOB_OUT_DIR:-<not set>}"
echo "LOCAL_JOB_OUT_DIR: ${LOCAL_JOB_OUT_DIR:-<not set>}"
echo "CONTAINER_IMAGE_PATH: ${CONTAINER_IMAGE_PATH:-<not set>}"
echo "ENV_SETUP_SCRIPT: ${ENV_SETUP_SCRIPT:-<not set>}"
echo "NODES: ${NODES:-<not set>}"
echo "GPUS_PER_NODE: ${GPUS_PER_NODE:-<not set>}"
echo "INSTALL_PACKAGE: ${INSTALL_PACKAGE:-<not set>}"
echo "UPDATE_PYTHONPATH: ${UPDATE_PYTHONPATH:-<not set>}"

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
    rsync --delete-after --filter=":e- .gitignore" --filter="- .git/" -a ./ "${TARGET_SOURCE_DIR}"
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
make_env_vars ${TRANSFER_VARS} CUDA_LAUNCH_BLOCKING TORCHINDUCTOR_COORDINATE_DESCENT_TUNING \
  TORCHINDUCTOR_COORDINATE_DESCENT_CHECK_ALL_DIRECTIONS TORCHINDUCTOR_COORDINATE_DESCENT_RADIUS \
  PYTORCH_CUDA_ALLOC_CONF TORCHINDUCTOR_AUTOTUNE_IN_SUBPROC TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS

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
kubectl -n "${VOLCANO_NAMESPACE}" wait --for=condition=Ready "${POD_POD_NAME}" --timeout=100m

# Tail logs from a specific container (e.g., "trainer") in all pods of this job
for P in $(kubectl get pods -l volcano.sh/job-name="$VCJOB_NAME" -o name); do
  echo "=== $P ==="
  kubectl logs -f "$P" &
done
wait

