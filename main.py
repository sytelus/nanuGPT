from grokking.config import Config
from grokking.train import train


if __name__ == "__main__":
    config = Config(default_config_filepath='configs/grok_baseline.yaml')
    train(config)
