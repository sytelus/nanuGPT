from gptplay.config import Config
from gptplay import trainer


if __name__ == "__main__":
    # specify config file to use as first argument in commandline
    config = Config(default_config_filepath='configs/grokking/prime223.yaml')
    trainer.train(config)
