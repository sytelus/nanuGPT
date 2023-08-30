import os

from grokking.config import Config
from grokking.train import train

from grokking.logger import Logger, DEFAULT_WANDB_METRICS
from grokking import utils

if __name__ == "__main__":
    config = Config(default_config_filepath='../configs/grok_baseline.yaml')

    out_dir = utils.full_path(config['out_dir'], create=True)

    logger = Logger(log_filepath=os.path.join(out_dir, 'seed_search_wd0_magic8.txt'),
                    project=config['wandb_project'], run_name="magic8_seed_search_wd0",
                    run_description="Find the distribution of seed that works well with data loader seed 8 but with weight decay = 0",
                    enable_wandb=config['use_wandb'], master_process=True,
                    project_config=config,
                    wandb_metrics=DEFAULT_WANDB_METRICS + [
                        {"name": "train/acc", "step_metric":"train/step", "summary":"max", "goal":"max"},
                        {"name": "val/acc", "step_metric":"train/step", "summary":"max", "goal":"max"},
                        {"name": "lr", "step_metric":"train/step", "summary":"max", "goal":"max"},
                        {"name": "ETA_hr", "step_metric":"train/step", "summary":"max", "goal":"max"},
                        {"name": "w_norm", "step_metric":"train/step", "summary":"min", "goal":"min"},
                        {"name": "train/d_loss", "step_metric":"train/step", "summary":"min", "goal":"min"},
                        {"name": "val/d_loss", "step_metric":"train/step", "summary":"min", "goal":"min"},
                        {"name": "train/ewa_loss", "step_metric":"train/step", "summary":"min", "goal":"min"},
                        {"name": "val/ewa_loss", "step_metric":"train/step", "summary":"min", "goal":"min"},
                        {"name": "w_norm_ewa", "step_metric":"train/step", "summary":"min", "goal":"min"},
                    ])

    # for i in range(20000):
    #     config['seed'] = i
    train(config, logger)

    logger.finish()
