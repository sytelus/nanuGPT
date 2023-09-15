import random

from gptplay.config import Config
from gptplay.train import train

from gptplay import logging
from gptplay import utils

if __name__ == "__main__":
    config = Config(default_config_filepath='configs/grokking/baseline.yaml')

    logger = logging.Logger(master_process=True, **config['logging'])
    logger.log_sys_info()
    logger.log_config(config)

    for i in range(1000):
        config['general']['seed'] = random.randint(0, 32000)
        config['data']['data_loader_seed'] = random.randint(0, 32000)
        train(config, logger)

    logger.all_done()