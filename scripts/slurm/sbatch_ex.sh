#! /bin/bash

####################################################################################################
# This script submits a job to slurm cluster
# It copies the source directory to a shared location and submits a script that will
# run srun to launch processes. The reason we need this script is because srun runs
# synchronously while sbatch exits after submission. This script will typically run
# from the login node in slurm environment which may not have GPUs or other resources.
####################################################################################################


set -eu -o pipefail # fail if any command failes, log all commands, -o xtrace

SOURCE_DIR=${SOURCE_DIR:-.} # where is source directory
export JOB_NAME=${JOB_NAME:-test_job}
export START_COMMAND=${START_COMMAND:-"$@"}
export DATA_ROOT=${DATA_ROOT:-} # data directory to mount in container
export OUT_DIR=${OUT_DIR:-"${HOME}/out_dir"} # set default output directory

NODES=${NODES:-1}
MAX_GPUS_PER_NODE=${MAX_GPUS_PER_NODE:-8}
export GPUS_PER_NODE="${GPUS_PER_NODE:-${MAX_GPUS_PER_NODE}}"

PARTITION=${PARTITION:-}
RESERVATION=${RESERVATION:-}
export CONTAINER_IMAGE_PATH=${CONTAINER_IMAGE_PATH:-"pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel"} #docker://@nvcr.io#nvidia/pytorch:24.07-py3
export ENV_SETUP_SCRIPT=${ENV_SETUP_SCRIPT:-} # script to setup environment for specific cluster

export INSTALL_PACKAGE=${INSTALL_PACKAGE:-1} # pip install in source directory
export UPDATE_PYTHONPATH=${UPDATE_PYTHONPATH:-0} # add source dir to PYTHONPATH (ignored if INSTALL_PACKAGE=1)
RESTARTABLE=${RESTARTABLE:-1}   # is job restartable if preempted?

SCRIPT_DIR="$(dirname "$(realpath "$0")")"

export JOB_OUT_DIR="${OUT_DIR}/${JOB_NAME}/$(date +%Y-%m-%d_%H-%M-%S_%3N)" # append job info
mkdir -p "${JOB_OUT_DIR}"

PARTITION_ARG=""
RESERVATION_ARG=""
REQUEUE_ARG=""
if [ ! -z "${PARTITION:-}" ]; then
    PARTITION_ARG="--partition=${PARTITION}"
fi
if [ ! -z "${RESERVATION:-}" ]; then
    RESERVATION_ARG="--reservation=${RESERVATION}"
fi
if [ "${GPUS_PER_NODE}" -lt "${MAX_GPUS_PER_NODE}" ]; then
    SHARE_NODE_ARG="--oversubscribe"
else
    SHARE_NODE_ARG="--overcommit --exclusive"
fi
if [ "${RESTARTABLE}" -eq 1 ]; then
    REQUEUE_ARG="--requeue"
fi

# output core variables so user can see what is being used
echo "SOURCE_DIR: $(realpath ${SOURCE_DIR:-<not set>})"
echo "START_COMMAND: ${START_COMMAND:-<not set>}"
echo "JOB_OUT_DIR: ${JOB_OUT_DIR:-<not set>}"
echo "DATA_ROOT: ${DATA_ROOT:-<not set>}"
echo "CONTAINER_IMAGE_PATH: ${CONTAINER_IMAGE_PATH:-<not set>}"
echo "ENV_SETUP_SCRIPT: ${ENV_SETUP_SCRIPT:-<not set>}"
echo "RESTARTABLE: ${RESTARTABLE:-<not set>}"
echo "NODES: ${NODES:-<not set>}"
echo "GPUS_PER_NODE: ${GPUS_PER_NODE:-<not set>}"
echo "PARTITION: ${PARTITION:-<not set>}"
echo "RESERVATION: ${RESERVATION:-<not set>}"

if [ ! -z "${ENV_SETUP_SCRIPT}" ]; then
    # copy to job out dir so its available in the container
    cp "${ENV_SETUP_SCRIPT}" "${JOB_OUT_DIR}/env_setup.sh"
    # update the variable to point to the copied file
    export JOB_ENV_SETUP_SCRIPT="${JOB_OUT_DIR}/env_setup.sh"
fi

# All tasks in job works off of the same source directory
# We wll also install package requirements from this directory
export TARGET_SOURCE_DIR="${JOB_OUT_DIR}/source_dir"
rm -rf "${TARGET_SOURCE_DIR}"
mkdir -p "${TARGET_SOURCE_DIR}"
pushd "${SOURCE_DIR}"
if git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
    git archive --format=tar HEAD | tar -x -C "${TARGET_SOURCE_DIR}"
else
    rsync -a --exclude='.git' ./ "${TARGET_SOURCE_DIR}"
fi
popd

# copy slurm scripts to job out dir so we can use it in cluster
export SLURM_SCRIPT_DIR="${JOB_OUT_DIR}/slurm_scripts"
# copy all slurm scripts to job out dir so we can use them in cluster
rm -rf "${SLURM_SCRIPT_DIR}"
mkdir -p "${SLURM_SCRIPT_DIR}"
cp "${SCRIPT_DIR}/"*.sh "${SLURM_SCRIPT_DIR}/"
chmod +x "${SLURM_SCRIPT_DIR}/"*.sh

sbatch \
    ${PARTITION_ARG} ${RESERVATION_ARG} ${SHARE_NODE_ARG} ${REQUEUE_ARG} \
    --nodes=${NODES} \
    --gpus-per-node=${GPUS_PER_NODE} \
    --job-name=${JOB_NAME} \
    --output="${JOB_OUT_DIR}/sbatch_log.txt" \
    --error="${JOB_OUT_DIR}/sbatch_err.txt" \
    --mem=0 \
    "${SLURM_SCRIPT_DIR}/srun_ex.sh" # this will be run once on primary node

echo "Job submitted. Use for job status:"
echo "tail -f '${JOB_OUT_DIR}/srun_err_000.txt'"
