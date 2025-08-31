#! /bin/bash

set -eu -o pipefail # fail if any command failes, log all commands, -o xtrace

# run this from project root
# setup below values

# need to declare separarely because we are using it as parameter to submit script
export JOB_NAME=nanugpt-test

# directory where this script is running
SCRIPT_DIR="$(dirname "$(realpath "$0")")"

NODES=1 \
CONTAINER_IMAGE_PATH=nvcr.io/nvidia/pytorch:25.08-py3 \
OUT_DIR=/data/shitals \
DATA_ROOT=/data/shitals/data \
bash ${SCRIPT_DIR}/volcano_submit.sh --general.project_name ${JOB_NAME} $@