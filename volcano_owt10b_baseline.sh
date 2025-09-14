#! /bin/bash

set -eu -o pipefail # fail if any command failes, log all commands, -o xtrace

# run this from project root
# setup below values

# can't inline these vars because we are using it as parameter to submit script
export RUN_NAME=owt-10k-karpathy-llmc
export RUN_DESC="Baseline: Keller Model+Scaler+WSD+LR3X 10.666B tokens"
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
    --general.run_name "${RUN_NAME}" \
    --general.run_description "${RUN_DESC}"