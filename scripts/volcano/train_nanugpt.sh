#! /bin/bash

set -eu -o pipefail # fail if any command failes, log all commands, -o xtrace

# run this from project root
# setup below values

# can't inline these vars because we are using it as parameter to submit script
export RUN_NAME=owt-full-baseline-b60
export JOB_NAME=gpt-std

# directory where this script is running
SCRIPT_DIR="$(dirname "$(realpath "$0")")"

export OUT_DIR=/data/shitals
export DATA_ROOT=/data/shitals/data
export DEVICE_BATCH_SIZE=60 # 60 for 192GB, 12 for 80GB
export TRANSFER_VARS="DATA_ROOT WANDB_API_KEY WANDB_HOST"

NODES=1 \
bash ${SCRIPT_DIR}/vsubmit.sh train.py configs/train_gpt2/openwebtext.yaml --general.project_name nanugpt-openwebtext --general.run_name ${RUN_NAME} --training.device_batch_size ${DEVICE_BATCH_SIZE}