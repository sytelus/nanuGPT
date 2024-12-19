#! /bin/bash

# this script starts the worker script in slurm

set -eu -o xtrace -o pipefail # fail if any command failes, log all commands

# required and optional variable
REQUIRED_VARS=("START_SCRIPT" "GPUS_PER_NODE" "TARGET_SOURCE_DIR")
USE_TORCHRUN=${USE_TORCHRUN:-0} # launcher to use worker processes
START_SCRIPT_ARGS=${START_SCRIPT_ARGS:-}    # arguments to pass to the entry script
MASTER_PORT=${MASTER_PORT:-} # port to use for torchrun
BIND_CORES=${BIND_CORES:-}  # cores allowed to bind launch script to
SLURM_LAUNCH_NODE_IPADDR=${SLURM_LAUNCH_NODE_IPADDR:-localhost} # should be set by slurm
PACKAGE_INSTALL_DIR=${PACKAGE_INSTALL_DIR:-}    # path where package and its dependencies are installed

### ---------- Check required environment variables
for var in "${REQUIRED_VARS[@]}"; do
    [ -z "${!var}" ] && { echo "Error: Required environment variable '$var' is not set." >&2; exit 1; }
done
### ---------- End check required environment variables

# add the package install directory to the python path
# this directory is shared between all nodes
if [ ! -z "$PACKAGE_INSTALL_DIR" ]; then
    export PYTHONPATH=$PACKAGE_INSTALL_DIR:$PYTHONPATH
fi

cd "${TARGET_SOURCE_DIR}"

# normally don't use torchrun as it doesn't scale as well as direct launch
if [ "${USE_TORCHRUN}" = "1" ]; then
    # build the torchrun command
    TORCH_RUN_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $SLURM_JOB_NUM_NODES --node_rank $SLURM_NODEID --master_addr $SLURM_LAUNCH_NODE_IPADDR"
    if [ -n "${MASTER_PORT}" ]; then
        TORCH_RUN_ARGS="${TORCH_RUN_ARGS} --master_port $MASTER_PORT"
    fi
    eval "torchrun $TORCH_RUN_ARGS $START_SCRIPT $START_SCRIPT_ARGS"
else
    # setup vars needed for torch.dist.init_process_group
    export CUDA_VISIBLE_DEVICES=${SLURM_LOCALID}
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
