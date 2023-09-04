import os
import time

from gptplay.config import Config
from gptplay.train import train

from gptplay import utils

if __name__ == "__main__":
    config = Config(default_config_filepath='configs/grokking/baseline.yaml')

    train(config)

    logger.all_done()