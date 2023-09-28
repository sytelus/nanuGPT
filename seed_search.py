import random
from datetime import datetime

from gptplay.config import Config
from gptplay.trainer import train

from gptplay import logging

if __name__ == "__main__":
    config = Config(default_config_filepath='configs/grokking/prime223.yaml')

    config['logging']['enable_summaries'] = False
    config['eval']['eval_every'] = 1000000 # only eval the last step
    config['eval']['save_checkpoint'] = False
    config['training']['num_steps'] = 3000 # only train for 1M steps
    config['logging']['allow_overwrite_log'] = False
    config['logging']['log_filename'] = 'seed_search_' + datetime.now().strftime('%Y%m%d%H%M%S') + '.log'
    config['training']['enable_train_log'] = False

    logging_config = config['logging']
    logger = logging.Logger(master_process=True, **logging_config)


    for i in range (120):
        config['general']['seed'] = random.randint(0, 32000)
        config['data']['data_loader_seed'] = random.randint(0, 32000)
        train(config, logger=logger)

    logger.all_done()