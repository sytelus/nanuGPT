#!/usr/bin/env python3

# usage: python train.py <config_file>
# example: python train.py configs/train_gpt2/tinyshakespeare.yaml
# distributed training: torchrun --standalone --nproc_per_node=8 train.py configs/train_gpt2/tinyshakespeare.yaml
# or use provided shell scripts in root directory

# this is the main training script that instantiates the model, dataset
# optimizer, lr scheduler etc according to the provided config file
# and runs the training loop

from nanugpt.train import train
from nanugpt.config import Config

if __name__ == "__main__":
    # specify config file to use as first argument in commandline
    config = Config()
    train(config)