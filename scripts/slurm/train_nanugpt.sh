#! /bin/bash

set -eu -o pipefail # fail if any command failes, log all commands, -o xtrace

# run this from project root
# setup below values

export  JOB_NAME=<my_job_name>

# NODES=1 \
# PARTITION=<my_partition> \
# RESERVATION=<my_reservation> \
# DATA_ROOT="/mnt/path/to/data" \
# OUT_DIR=<my_out_dir> \
bash sbatch_ex.sh train.py --general.project_name ${JOB_NAME} $@