from nanugpt import glogging as logging
from nanugpt import common
from nanugpt import utils
from nanugpt.config import Config
from nanugpt.tokenize_dataset import tokenize

if __name__ == "__main__":
    # specify config file to use as first argument in commandline
    config = Config(default_config_filepath='configs/tokenize/tiktoken_gpt2.yaml')

    logger = common.setup_logger(config=config, is_master=utils.is_master_process())

    data_config = config['data']
    tokenizer_config = config['tokenizer']

    get_tokenizer_factory = utils.import_fn(tokenizer_config['module'])
    tokenizer_factory = get_tokenizer_factory(**tokenizer_config['module_kwargs'])

    tokenize(tokenizer_factory=tokenizer_factory, **data_config)

    logging.shutdown()