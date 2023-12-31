#!/bin/bash
#fail if any errors
set -e
set -o xtrace

torchrun --standalone --nproc_per_node=$(python -c "import torch; print(torch.cuda.device_count())") nanugpt/train.py configs/train_gpt2/tinyshakespeare.yaml