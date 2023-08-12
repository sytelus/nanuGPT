import os

from grokking.config import Config
from grokking.train import train

from grokking.logger import Logger
from grokking import utils

if __name__ == "__main__":
    config = Config(default_config_filepath='../configs/grok_baseline.yaml')

    out_dir = utils.full_path(config['out_dir'], create=True)

    logger = Logger(log_filepath=os.path.join(out_dir, 'high_val_e50.txt'),
                    enable_wandb=config['use_wandb'], master_process=True,
                    wandb_project=config['wandb_project'], wandb_run_name=config['wandb_run'],
                    config=config,
                    )

    train(config, logger)

    logger.finish()
