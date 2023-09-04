import os
import time

from gptplay.config import Config
from gptplay.train import train

from gptplay import logging
from gptplay import utils

if __name__ == "__main__":
    config = Config(default_config_filepath='configs/grokking/baseline.yaml')

    logger = logging.Logger(master_process=True, **config['logging'])
    logger.log_sys_info()
    logger.log_config(config)

    for i in range(20000):
        config['general']['seed'] = i
        train(config, logger)

    logger.all_done()