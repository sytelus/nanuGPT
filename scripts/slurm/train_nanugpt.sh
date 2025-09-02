#! /bin/bash

set -eu -o pipefail # fail if any command failes, log all commands, -o xtrace

# run this from project root
# setup below values

export  JOB_NAME=${JOB_NAME:-nanugpt_test}

# directory where this script is running
SCRIPT_DIR="$(dirname "$(realpath "$0")")"

# NODES=1 \
# PARTITION=<my_partition> \
# RESERVATION=<my_reservation> \
# DATA_ROOT="/mnt/path/to/data" \
# OUT_DIR=<my_out_dir> \
NODES=4 bash ${SCRIPT_DIR}/sbatch_ex.sh train.py configs/train_gpt2/openwebtext.yaml --general.project_name ${JOB_NAME} $@