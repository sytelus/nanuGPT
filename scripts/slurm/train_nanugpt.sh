#! /bin/bash

set -eu -o xtrace -o pipefail # fail if any command failes, log all commands

# run this from project root
# setup below values

# NODES=1 \
# PARTITION=<my_partition> \
# RESERVATION=<my_reservation> \
# DATA_ROOT="/mnt/path/to/data" \
# JOB_NAME=<my_job_name> \
# OUT_DIR=<my_out_dir> \
bash sbatch_ex.sh train.py --general.project_name ${JOB_NAME} $@