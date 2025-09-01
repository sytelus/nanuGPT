#! /bin/bash

set -eu -o pipefail # fail if any command failes, log all commands, -o xtrace

# run this from project root
# setup below values

# need to declare separarely because we are using it as parameter to submit script
export JOB_NAME=nanugpt-openwebtext-tokens10b-keller_adamw

# directory where this script is running
SCRIPT_DIR="$(dirname "$(realpath "$0")")"

NODES=1 \
OUT_DIR=/data/shitals \
DATA_ROOT=/data/shitals/data \
bash ${SCRIPT_DIR}/volcano_submit.sh train.py configs/train_gpt2/openwebtext_tokens10b_keller_adamw.yaml --general.project_name nanugpt-openwebtext --general.run_name ${JOB_NAME} $@