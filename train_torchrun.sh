#! /bin/bash
set -eu -o pipefail # -o xtrace # fail if any command failes, log all commands, -o xtrace

export OUT_DIR=${OUT_DIR:-/data/${USER}} # base output directory where we will create sub dir for this run

# make sure OUT_DIR is set (it doesn't need to exist yet)
if [ -z "${OUT_DIR:-}" ]; then
  echo "Please set OUT_DIR env var to point to directory where output will be stored."
  exit 1
else
  echo "OUT_DIR is $OUT_DIR"
fi

# check DATA_ROOT is set and exists
if [ -z "${DATA_ROOT:-}" ]; then
  echo "Please set DATA_ROOT env var to point to directory where data is stored."
  exit 1
elif [ ! -d "$DATA_ROOT" ]; then
  echo "DATA_ROOT directory $DATA_ROOT does not exist. Please create it first or set DATA_ROOT env var to an existing directory."
  exit 1
else
  echo "DATA_ROOT is $DATA_ROOT"
fi

export NPROC_PER_NODE=${NPROC_PER_NODE:-$(python -c "import torch; print(torch.cuda.device_count())")}
export NODES=${NODES:-1}
export RANK=${RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-"localhost"}
export MASTER_PORT=${MASTER_PORT:-23456} # can be any free port
export START_COMMAND=${START_COMMAND:-"$@"} # use all args to this script as command we will execute
if [ -z "$START_COMMAND" ]; then
  echo "No command to run. Invoke this script as:"
  echo "./train_dist.sh train.py configs/train_gpt2/openwebtext.yaml"
  exit 1
fi

# if DEBUG_WITH_CUDA_GDB is set then run command under cuda-gdb
if [ -n "${DEBUG_WITH_CUDA_GDB:-}" ]; then
  torchrun --nproc_per_node=${NPROC_PER_NODE} --nnodes=${NODES} --node_rank=${RANK} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} --no_python scripts/misc/cude_gdb_worker.sh
else
  torchrun --nproc_per_node=${NPROC_PER_NODE} --nnodes=${NODES} --node_rank=${RANK} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} ${START_COMMAND}
fi

