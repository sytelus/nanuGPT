#! /bin/bash
set -eu -o pipefail # -o xtrace # fail if any command failes, log all commands, -o xtrace

# simpler command
# torchrun --standalone --nproc_per_node=$(python -c "import torch; print(torch.cuda.device_count())") nanugpt/train.py configs/train_gpt2/tinyshakespeare.yaml

export OUT_DIR=${OUT_DIR:-/data/${USER}} # base output directory where we will create sub dir for this run

# make sure OUT_DIR exists already or exit
if [ ! -d "$OUT_DIR" ]; then
  echo "Output directory $OUT_DIR does not exist. Please create it first or set OUT_DIR env var to an existing directory."
  exit 1
else
  echo "Output directory is $OUT_DIR"
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
  export CUDA_LAUNCH_BLOCKING=1
  export TORCH_COMPILE_DEBUG=1
  #export TORCH_LOGS=output_code
  export TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1

  # Tiny wrapper so rank 0 is debugged, others just exec the program
  cat >/tmp/worker_cmd.sh <<'EOS'
#!/usr/bin/env bash
set -euo pipefail

# switch to current working directory
cd "$(pwd)"

if [[ "${LOCAL_RANK:-0}" == "0" ]]; then
  exec cuda-gdb -ex 'set pagination off' -ex 'set cuda break_on_launch application' -ex 'set cuda api_failures stop' --args exec python -X faulthandler ${START_COMMAND}
else
  exec python -X faulthandler -u ${START_COMMAND}
fi
EOS

  chmod +x /tmp/worker_cmd.sh

  torchrun --nproc_per_node=${NPROC_PER_NODE} --nnodes=${NODES} --node_rank=${RANK} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} --no_python /tmp/worker_cmd.sh
else
  torchrun --nproc_per_node=${NPROC_PER_NODE} --nnodes=${NODES} --node_rank=${RANK} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} ${START_COMMAND}
fi

