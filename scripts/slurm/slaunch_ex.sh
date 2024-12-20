#! /bin/bash

####################################################################################################
# This script is the final stage of submission to the slurm cluster. This script is spawned per
# worker process and is needed as wrapper to the actual start script to setup environment variables
# that will be used by torch.distributed.init_process_group. Another additional setup this script
# does is to update PYTHONPATH to include shared package install directory.
####################################################################################################

set -eu -o xtrace -o pipefail # fail if any command failes, log all commands

# required and optional variable
REQUIRED_VARS=("START_SCRIPT" "GPUS_PER_NODE" "TARGET_SOURCE_DIR")
USE_TORCHRUN=${USE_TORCHRUN:-0} # launcher to use worker processes
START_SCRIPT_ARGS=${START_SCRIPT_ARGS:-}    # arguments to pass to the entry script
MASTER_PORT=${MASTER_PORT:-} # port to use for torchrun
BIND_CORES=${BIND_CORES:-}  # cores allowed to bind launch script to
SLURM_LAUNCH_NODE_IPADDR=${SLURM_LAUNCH_NODE_IPADDR:-localhost} # should be set by slurm
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
export MASTER_ADDR="${MASTER_ADDR:-$SLURM_LAUNCH_NODE_IPADDR}"
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
if [[ "${gpu_array[$SLURM_LOCALID]}" -eq "${min_gpu_id}" ]]; then
    export IS_NODE_PROC0=1
else
    export IS_NODE_PROC0=0
fi

cd "${TARGET_SOURCE_DIR}"

# package installation if requested
if [ "${INSTALL_PACKAGE}" = "1" ]; then
    # create temp dir to store marker file
    temp_dir="${TMPDIR:-/tmp}/slaunch_package_marker"
    marker_file="${temp_dir}/install_done"
    mkdir -p "$temp_dir"

    # if head process in the node then install package
    if [[ "$IS_NODE_PROC0" -eq 1 ]]; then
        rm -f "$marker_file"
        pip install --no-cache-dir --editable "${TARGET_SOURCE_DIR}"

        # Create the marker file to signal other workers
        touch "$marker_file"
    else
        # block other workers until head process is done installing package
        echo "Waiting for marker file to be created..."
        inotifywait -q -e create --format "%f" "$temp_dir" | while read file; do
            if [[ "$file" == "$(basename "$marker_file")" ]]; then
                break
            fi
        done
    fi
else
    # add the package to PYTHONPATH so it acts like installed package
    if [ "${UPDATE_PYTHONPATH}" = "1" ]; then
        export PYTHONPATH="${TARGET_SOURCE_DIR}":${PYTHONPATH:-}
    fi
fi

# normally don't use torchrun as it doesn't scale as well as direct launch
if [ "${USE_TORCHRUN}" = "1" ]; then
    # build the torchrun command
    TORCH_RUN_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $SLURM_JOB_NUM_NODES --node_rank $SLURM_NODEID --master_addr $SLURM_LAUNCH_NODE_IPADDR --master_port $MASTER_PORT"
    eval "torchrun $TORCH_RUN_ARGS $START_SCRIPT $START_SCRIPT_ARGS"
else
    # setup vars needed for torch.dist.init_process_group
    export CUDA_VISIBLE_DEVICES=${gpu_array[$SLURM_LOCALID]}
    export NODE_RANK=$SLURM_NODEID
    export RANK=$SLURM_PROCID # global rank
    export LOCAL_RANK=$SLURM_LOCALID # rank within the node, not used by init_process_group but we set it for consistency
    export WORLD_SIZE=$((SLURM_NNODES * SLURM_NTASKS_PER_NODE))

    if [ -n "${BIND_CORES}" ]; then # restrict workers to specific cores is requested
        exec taskset -c $BIND_CORES python -u "${START_SCRIPT}" ${START_SCRIPT_ARGS}
    else
        exec python -u "${START_SCRIPT}" ${START_SCRIPT_ARGS}
    fi
fi
