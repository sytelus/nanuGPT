#! /bin/bash

# run this from project root

SCRIPT_DIR="$(dirname "$(realpath "$0")")"

export START_SCRIPT="train.py"
export START_SCRIPT_ARGS="configs/train_gpt2/openwebtext.yaml"

# SETUP BELOW VALUES!!

# export CONTAINER_IMAGE_PATH="/mnt/path/to/image.sif"
# export CONTAINER_MOUNTS="/mnt/path/to/data:/mnt/path/to/data"
# export ENV_SETUP_SCRIPT="/mnt/path/to/env_setup.sh"
# export DATA_ROOT="/mnt/path/to/data"

# PARTITION=<my_partition> \
# RESERVATION=<my_reservation> \
RESTARTABLE=0 \
bash "$SCRIPT_DIR/sbatch_ex.sh"
