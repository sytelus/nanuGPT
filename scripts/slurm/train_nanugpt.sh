#! /bin/bash

set -eu -o xtrace -o pipefail # fail if any command failes, log all commands

# run this from project root
# NODES=4 DATA_ROOT=<my data dir> bash ./scripts/slurm/sbatch_ex.sh configs/train_gpt2/openwebtext.yaml



# !! setup below values
# DATA_ROOT is mounted inside the container at the same location, if not empty
# PARTITION=<my_partition> \
# RESERVATION=<my_reservation> \
# DATA_ROOT="/mnt/path/to/data" \
# look for other available params in sbatch_ex.sh
bash "$SCRIPT_DIR/sbatch_ex.sh" "$@"
