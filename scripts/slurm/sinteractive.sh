#! /bin/bash

set -eu -o xtrace -o pipefail # fail if any command failes, log all commands

# required and optional variable
REQUIRED_VARS=("GPUS_PER_NODE" "CONTAINER_IMAGE_PATH")
CONTAINER_MOUNTS=${CONTAINER_MOUNTS:-}  # app specific mounts to be attached to container as source:destination
SYS_CONTAINER_MOUNTS=${SYS_CONTAINER_MOUNTS:-}  # system specific mounts to be attached to container as source:destination
JOB_ENV_SETUP_SCRIPT=${JOB_ENV_SETUP_SCRIPT:-} # script to setup environment for specific cluster
NODES=${NODES:-1}
PARTITION=${PARTITION:-}
RESERVATION=${RESERVATION:-}
MAX_GPUS_PER_NODE=${MAX_GPUS_PER_NODE:-8}
GPUS_PER_NODE="${GPUS_PER_NODE:-${MAX_GPUS_PER_NODE}}"
NODE_LIST=${NODE_LIST:-}
OUT_DIR=${OUT_DIR:-"${HOME}/out_dir"} # set default output directory
JOB_NAME=${JOB_NAME:-'slurm_interactive_job'}
export INTERACTIVE_JOB=1    # used by env_setup.sh to setup environment for interactive job


### ---------- Check required environment variables
for var in "${REQUIRED_VARS[@]}"; do
    [ -z "${!var}" ] && { echo "Error: Required environment variable '$var' is not set." >&2; exit 1; }
done
### ---------- End check required environment variables

export JOB_OUT_DIR="${OUT_DIR}/${JOB_NAME}/$(date +%Y-%m-%d_%H-%M-%S_%3N)" # append job info
mkdir -p "${JOB_OUT_DIR}"

# setup cluster specific environment variables, these will be inherited by the container
if [ ! -z "$JOB_ENV_SETUP_SCRIPT" ]; then
    source "$JOB_ENV_SETUP_SCRIPT"
fi


PARTITION_ARG=""
RESERVATION_ARG=""
NODELIST_ARG=""
if [ ! -z "${PARTITION:-}" ]; then
    PARTITION_ARG="--partition=${PARTITION}"
fi
if [ ! -z "${RESERVATION:-}" ]; then
    RESERVATION_ARG="--reservation=${RESERVATION}"
fi
if [ ! -z "${NODE_LIST}" ]; then
    NODELIST_ARG="--nodelist=${NODE_LIST}"
fi
if [ "${GPUS_PER_NODE}" -lt "${MAX_GPUS_PER_NODE}" ]; then
    SHARE_NODE_ARG="--oversubscribe"
else
    SHARE_NODE_ARG="--overcommit --exclusive"
fi

ALL_CONTAINER_MOUNTS="${JOB_OUT_DIR}:${JOB_OUT_DIR}"
if [ ! -z "${CONTAINER_MOUNTS}" ]; then
    ALL_CONTAINER_MOUNTS="${ALL_CONTAINER_MOUNTS},${CONTAINER_MOUNTS}"
fi
if [ ! -z "${SYS_CONTAINER_MOUNTS}" ]; then
    ALL_CONTAINER_MOUNTS="${ALL_CONTAINER_MOUNTS},${SYS_CONTAINER_MOUNTS}"
fi

srun ${PARTITION_ARG} ${RESERVATION_ARG} ${SHARE_NODE_ARG} ${NODELIST_ARG} \
    --nodes=${NODES} \
    --gpus-per-node=${GPUS_PER_NODE} \
    -o "${JOB_OUT_DIR}/srun_log_interactive.txt" \
    -e "${JOB_OUT_DIR}/srun_err_interactive.txt" \
    --container-image "${CONTAINER_IMAGE_PATH}" \
    --container-mounts "${ALL_CONTAINER_MOUNTS}" \
    --container-writable --no-container-remap-root \
    --wait=60 --kill-on-bad-exit=1 --label \
    --task-epilog="bash -c '/usr/bin/pkill -U \$SLURM_JOB_USER sshd'"
    --pty /bin/bash -i
