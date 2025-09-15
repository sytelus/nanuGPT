#! /bin/bash

set -eu -o pipefail # fail if any command failes, log all commands, -o xtrace

# Runs Keller Jordan's run with AdamW, model treaks, scaler, LE and WSD schedule for 10.7B tokens.
# This does not include every twek he made for the record.
# run this from project root
# setup below values

# can't inline these vars because we are using it as parameter to submit script
export RUN_NAME=owt-10b-keller-adamw-g1024-l128-lr26e-4-warm256-wd-half-no-scaler
export RUN_DESC="global bz 1024, device bz 128, LR 25.5e-4  Keller Model+AdamW+WSD 10B tokens"
export TORCHINDUCTOR_COORDINATE_DESCENT_TUNING=1
# do not optimize scaler outputs on compile, will cause graph break warnings
export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1

export JOB_NAME=gpt-std

# directory where this script is running
SCRIPT_DIR="$(dirname "$(realpath "$0")")"

export OUT_DIR=/data/shitals
export DATA_ROOT=/data/shitals/data
export TRANSFER_VARS="DATA_ROOT WANDB_API_KEY WANDB_HOST"

NODES=1 \
bash "${SCRIPT_DIR}/scripts/volcano/vsubmit.sh" train.py configs/train_gpt2/openwebtext_tokens10b_keller_adamw.yaml \
    --general.project_name nanugpt-owt10k \
    --training.global_batch_size 1024 --training.device_batch_size 128 --training.max_steps 10173 --optimizer.module_kwargs.learning_rate "25.452E-4" --scheduler.module_kwargs.warmup_iters 128 --optimizer.module_kwargs.weight_decay "0.05" \
    --general.run_name "${RUN_NAME}" \
    --general.run_description "${RUN_DESC}"