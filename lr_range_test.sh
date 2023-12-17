#!/bin/bash
#fail if any errors
set -e
set -o xtrace

python train.py "{$1};configs/hp_tuning/lr_range_test.yaml"