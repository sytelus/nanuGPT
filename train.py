import os
import time

from gptplay.config import Config
from gptplay.trainer import train

from gptplay import utils

if __name__ == "__main__":
    # specify config file to use as first argument in commandline
    config = Config(default_config_filepath='configs/grokking/prime223.yaml')
    train(config)
