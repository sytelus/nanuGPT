#! /bin/bash

####################################################################################################
# This script is the final stage of submission to the slurm cluster. This script is spawned per
# worker process and is needed as wrapper to the actual start script to setup environment variables
# that will be used by torch.distributed.init_process_group. Another additional setup this script
# does is to update PYTHONPATH to include shared package install directory.
####################################################################################################

set -eu -o pipefail # fail if any command failes, log all commands, -o xtrace

# required and optional variable
REQUIRED_VARS=("GPUS_PER_NODE" "TARGET_SOURCE_DIR")
USE_TORCHRUN=${USE_TORCHRUN:-0} # launcher to use worker processes
START_COMMAND=${START_COMMAND:-}    # command to run in slurm
MASTER_PORT=${MASTER_PORT:-} # port to use for torchrun
BIND_CORES=${BIND_CORES:-}  # cores allowed to bind launch script to
SLURM_LAUNCH_NODE_IPADDR=${SLURM_LAUNCH_NODE_IPADDR:-"localhost"} # should be set by slurm
INSTALL_PACKAGE=${INSTALL_PACKAGE:-0} # pip install in source directory
UPDATE_PYTHONPATH=${UPDATE_PYTHONPATH:-0} # add source dir to PYTHONPATH

### ---------- Check required environment variables
for var in "${REQUIRED_VARS[@]}"; do
    [ -z "${!var}" ] && { echo "Error: Required environment variable '$var' is not set." >&2; exit 1; }
done
### ---------- End check required environment variables

# master address and port is required for torch.distributed so communication with other nodes can happen
# SLURM_LAUNCH_NODE_IPADDR should be set but we create fallback to HEAD_NODE_IPADDR
SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-0}
export MASTER_ADDR="${MASTER_ADDR:-${SLURM_LAUNCH_NODE_IPADDR}}"
# Split GPU IDs into an array and find the minimum value
IFS=',' read -ra gpu_array <<< "${SLURM_STEP_GPUS}"
min_gpu_id=${gpu_array[0]}
for gpu_id in "${gpu_array[@]}"; do
    if (( gpu_id < min_gpu_id )); then
        min_gpu_id=$gpu_id
    fi
done
export MASTER_PORT=$((6000 + min_gpu_id))

# Setup the "head" process within the node
if [[ "${gpu_array[${SLURM_LOCALID}]}" -eq "${min_gpu_id}" ]]; then
    export IS_NODE_PROC0=1
else
    export IS_NODE_PROC0=0
fi

cd "${TARGET_SOURCE_DIR}"

# package installation if requested
if [ "${INSTALL_PACKAGE}" = "1" ]; then
    # create temp dir to store marker file
    LOCKDIR="${TMPDIR:-/tmp}/slaunch_package_fifo"
    mkdir -p "${LOCKDIR}"
    PIPEFILE="${LOCKDIR}/sync_pipe"
    DONEFILE="${LOCKDIR}/install_done"

    # Ensure the named pipe exists. If multiple processes attempt mkfifo simultaneously,
    # this might print an error once, but the pipe will still exist after the first success.
    if [[ ! -p "${PIPEFILE}" ]]; then
        (umask 000; mkfifo "${PIPEFILE}" 2>/dev/null || true)
    fi


    # if head process in the node then install package
    if [[ "$IS_NODE_PROC0" -eq 1 ]]; then
        # Writer process
        # Cleanup any old done file to ensure a fresh state for this run.
        rm -f "${DONEFILE}"

        # install the package in editable mode
        pip install -e .

        # Signal that initialization is done
        touch "${DONEFILE}"
        # Write a signal to the pipe and close it.
        # This will unblock all readers waiting on the pipe.
        echo "initialized" > "${PIPEFILE}"
        # Once we exit, the pipe closes and any reader that didn't get the line will get EOF.

        echo "Writer: Initialization complete."
        # If there are no readers, this still works fine. The writer doesn't block.
    else
        # Reader process
        if [ ! -f "${DONEFILE}" ]; then
            # Need to wait until initialization is done. We'll block on reading the pipe.
            if read line < "${PIPEFILE}"; then
                true # Do nothing. We just need to unblock the pipe.
            else
                # If we got here, it means the pipe was closed (EOF) before we got a line.
                # this will happen to all but one reader process.
                true # Do nothing. We just need to unblock the pipe.
            fi

            # After being unblocked, ensure done file is present.
            if [ ! -f "${DONEFILE}" ]; then
                echo "Unexpected behavior: Worker process with SLURM_PROCID=$SLURM_PROCID got unblocked while package install was still not done!" >&2   # >&2 redirects to stderr
                exit 1  # Non-zero exit code indicates error            fi
            fi
        fi
    fi
else
    # add the package to PYTHONPATH so it acts like installed package
    if [ "${UPDATE_PYTHONPATH}" = "1" ]; then
        export PYTHONPATH="${TARGET_SOURCE_DIR}:${PYTHONPATH:-}"
    fi
fi

# normally don't use torchrun as it doesn't scale as well as direct launch
if [ "${USE_TORCHRUN}" = "1" ]; then
    # build the torchrun command
    TORCH_RUN_ARGS="--nproc_per_node ${GPUS_PER_NODE} --nnodes ${SLURM_JOB_NUM_NODES} --node_rank ${SLURM_NODEID} --master_addr ${SLURM_LAUNCH_NODE_IPADDR} --master_port ${MASTER_PORT}"
    eval "OUT_DIR=\"${JOB_OUT_DIR}\" torchrun ${TORCH_RUN_ARGS} ${START_COMMAND}"
else
    # setup vars needed for torch.dist.init_process_group
    export CUDA_VISIBLE_DEVICES=${gpu_array[$SLURM_LOCALID]}
    export NODE_RANK=$SLURM_NODEID
    export RANK=$SLURM_PROCID # global rank
    export LOCAL_RANK=$SLURM_LOCALID # rank within the node, not used by init_process_group but we set it for consistency
    export WORLD_SIZE=$((SLURM_NNODES * SLURM_NTASKS_PER_NODE))
    export LOCAL_WORLD_SIZE=$((SLURM_NTASKS_PER_NODE))

    if [ -n "${BIND_CORES}" ]; then # restrict workers to specific cores is requested
        OUT_DIR="${JOB_OUT_DIR}" exec taskset -c $BIND_CORES python -u ${START_COMMAND}
    else
        OUT_DIR="${JOB_OUT_DIR}" exec python -u ${START_COMMAND}
    fi
fi
