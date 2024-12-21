#! /bin/bash

####################################################################################################
# This script submits uses srun to launch worker processes in the slurm cluster. The reason we need
# this script is because sbatch only runs script on master node once. If we need to launch multiple
# worker processes then we need to use srun. This script will typically run from the master node in
# slurm environment which may have GPUs or other resources. This script will first install the
# code as package in the container such that all dependencies are in persistent storage and then
# launch the target script on all nodes.
####################################################################################################

set -eu -o xtrace -o pipefail # fail if any command failes, log all commands

# required and optional variable
REQUIRED_VARS=("GPUS_PER_NODE" "CONTAINER_IMAGE_PATH" "JOB_OUT_DIR" "TARGET_SOURCE_DIR" "SLURM_SCRIPT_DIR")
CONTAINER_MOUNTS=${CONTAINER_MOUNTS:-}  # app specific mounts to be attached to container as source:destination
SYS_CONTAINER_MOUNTS=${SYS_CONTAINER_MOUNTS:-}  # system specific mounts to be attached to container as source:destination
JOB_ENV_SETUP_SCRIPT=${JOB_ENV_SETUP_SCRIPT:-} # script to setup environment for specific cluster
export USE_TORCHRUN=${USE_TORCHRUN:-0}  # use torchrun or direct slurm launch (recommanded)

# Some slurm environment use pmi or pmi2 plugins for MPI in which case set this
# to "--mpi=pmi2" or "--mpi=pmi". Default is to use whatever is configured.
# If using torchrun then this is overriden with "--mpi=none".
MPI_ARG=${MPI_ARG:-}


### ---------- Check required environment variables
for var in "${REQUIRED_VARS[@]}"; do
    [ -z "${!var}" ] && { echo "Error: Required environment variable '$var' is not set." >&2; exit 1; }
done
### ---------- End check required environment variables

SCRIPT_DIR="$(dirname "$(realpath "$0")")"

# setup cluster specific environment variables, these will be inherited by the container
if [ ! -z "${JOB_ENV_SETUP_SCRIPT}" ]; then
    source "${JOB_ENV_SETUP_SCRIPT}"
fi

# if we are restarting because of preempt then get the count
RESTART_COUNT=$(printf "%03d" ${SLURM_RESTART_COUNT:-0})

# override mpi arg if using torchrun as torchrun uses its own backend
if [ ${USE_TORCHRUN} -eq 0 ]; then
    MPI_ARG="--mpi=none"
fi

ALL_CONTAINER_MOUNTS="${JOB_OUT_DIR}:${JOB_OUT_DIR}"
if [ ! -z "${CONTAINER_MOUNTS}" ]; then
    ALL_CONTAINER_MOUNTS="${ALL_CONTAINER_MOUNTS},${CONTAINER_MOUNTS}"
fi
if [ ! -z "${SYS_CONTAINER_MOUNTS}" ]; then
    ALL_CONTAINER_MOUNTS="${ALL_CONTAINER_MOUNTS},${SYS_CONTAINER_MOUNTS}"
fi

NTASKS=$((SLURM_JOB_NUM_NODES * GPUS_PER_NODE))
# start the script in the container that will launch target script
srun --ntasks=${NTASKS} --ntasks-per-node=${GPUS_PER_NODE} ${MPI_ARG} \
    -o "${JOB_OUT_DIR}/srun_log_${RESTART_COUNT}.txt" \
    -e "${JOB_OUT_DIR}/srun_err_${RESTART_COUNT}.txt" \
    --container-image "${CONTAINER_IMAGE_PATH}" \
    --container-mounts "${ALL_CONTAINER_MOUNTS}" \
    --container-writable --no-container-mount-home --no-container-remap-root \
    --wait=60 --kill-on-bad-exit=1 --label \
    "${SLURM_SCRIPT_DIR}/slaunch_ex.sh"
