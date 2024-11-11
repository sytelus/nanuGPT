

from nanugpt import glogging as logging
from nanugpt.config import Config
from nanugpt.generate import Generator

if __name__ == "__main__":
    # specify config file to use as first argument in commandline
    # directory set in config file in generate/checkpoint_dir is used to load model
    config = Config(default_config_filepath='configs/train_gpt2/tinyshakespeare.yaml')
    logging_config = config['logging']
    logging_config['enable_wandb'] = False
    logger = logging.Logger(master_process=False, **logging_config)

    gen = Generator(config, logger)
    results = gen.generate(['\n'], 200)
    print(results)

    logger.shutdown()