import random
from datetime import datetime

from nanugpt.config import Config
from nanugpt.train import train

from nanugpt import glogging as logging

if __name__ == "__main__":
    config = Config(default_config_filepath='configs/grokking/prime223.yaml')

    config['logging']['summaries_stdout'] = False
    config['eval']['eval_every'] = 1000000 # only eval the last step
    config['eval']['save_checkpoint'] = False
    config['training']['max_steps'] = 3000 # only train for 1M steps
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