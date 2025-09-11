#!/usr/bin/env bash
# DON'T use -e here; we want to handle failures ourselves
set -uo pipefail

#############################################################
# This script is used torch_dist.sh.
#
# This script is a worker script to run under torchrun when
# DEBUG_WITH_CUDA_GDB is set. It runs rank 0 under cuda-gdb
# and other ranks normally, but keeps them alive if they fail.
# Invoke this script as:
# torchrun --nproc_per_node=... --nnodes=... --node_rank=... --master_addr=... --master_port=... --no_python scripts/misc/cude_g
# db_worker.sh
#############################################################


# stay in the caller's working dir
cd "$PWD"

export CUDA_LAUNCH_BLOCKING=1
export TORCH_COMPILE_DEBUG=1
#export TORCH_LOGS=output_code
export TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1

# Build the actual command: if START_COMMAND doesn't start with python, prefix it.
PY="${PYTHON_BIN:-python}"
CMD="${START_COMMAND}"
if [[ ! "${CMD}" =~ ^([[:alnum:]_./-]*python[[:digit:]\.]*)[[:space:]] ]]; then
    CMD="${PY} -u -X faulthandler ${CMD}"
fi

# portable "sleep forever"
sleep_forever() { while :; do sleep 2147483647 || sleep 3600; done; }

if [[ "${LOCAL_RANK:-0}" == "0" ]]; then
    # Rank 0 under cuda-gdb; ignore early TERM so we can hit breakpoints
    exec cuda-gdb -q \
        -ex "set pagination off" \
        -ex "set cuda break_on_launch none" \
        -ex "set cuda api_failures stop" \
        -ex "handle SIGTERM nostop noprint ignore" \
        -ex "handle SIGHUP nostop noprint ignore" \
        -ex set cuda memcheck off \
        -ex set cuda kernel_events none \
        -ex set cuda context_events none \
        -ex "run" \
        --args bash -lc "exec ${CMD}"
else
    # Optional: make peers less crashy (eager path)
    # export TORCHDYNAMO_DISABLE=1

    # Run peer; if it FAILS, DON'T exit—keep the process alive.
    set +e
    bash -lc "exec ${CMD}"
    rc=$?
    set -e
    if (( rc != 0 )); then
        echo "[peer LOCAL_RANK=${LOCAL_RANK}] exited rc=${rc}; keeping process alive for debugger." >&2
        # touch "/tmp/peer_${LOCAL_RANK}_crashed"
        # Optional: be nicer to collectives during debug
        export NCCL_BLOCKING_WAIT=1
        export NCCL_ASYNC_ERROR_HANDLING=0
        sleep_forever
    fi

    # If it succeeded, just exit normally
    exit 0
fi