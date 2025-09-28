#! /bin/bash

set -eu -o pipefail # fail if any command failes, log all commands, -o xtrace

# Runs Keller Jordan's run with AdamW, model treaks, scaler, LE and WSD schedule for 10.7B tokens.
# This does not include every twek he made for the record.
# run this from project root
# setup below values

# can't inline these vars because we are using it as parameter to submit script
export RUN_NAME=owt-10b-keller-muon
export RUN_DESC="Baseline: Keller Model+Scaler+WSD+Muon 10.666B tokens with Muon"
# export RUN_NAME=owt-160b-keller-muon
# export RUN_DESC="Baseline: Keller Model+Scaler+WSD+Muon 160B tokens with Muon"
export TORCHINDUCTOR_COORDINATE_DESCENT_TUNING=0
# do not optimize scaler outputs on compile, will cause graph break warnings
export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=0

export JOB_NAME=gpt-std

# directory where this script is running
SCRIPT_DIR="$(dirname "$(realpath "$0")")"

export OUT_DIR=/data/shitals
export DATA_ROOT=/data/shitals/data
export TRANSFER_VARS="DATA_ROOT WANDB_API_KEY WANDB_HOST"

# for 160B use --training.max_steps 325520 \

# To run original script use this:
# NODES=1 bash "${SCRIPT_DIR}/scripts/volcano/vsubmit.sh" scripts/alt_training/keller_train_gpt2_muon.py

NODES=1 \
bash "${SCRIPT_DIR}/scripts/volcano/vsubmit.sh" train.py configs/train_gpt2/openwebtext_tokens10b_keller_muon.yaml \
    --general.project_name nanugpt-owt10k \
    --general.run_name "${RUN_NAME}" \
    --general.run_description "${RUN_DESC}"
