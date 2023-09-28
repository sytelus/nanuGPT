import os

from gptplay import utils
from gptplay.config import Config
from gptplay.data import tokenize_hf_dataset

if __name__ == "__main__":
    # specify config file to use as first argument in commandline
    config = Config(default_config_filepath='configs/tokenize/tiktoken_gpt2.yaml')
    tokenization_config = config['tokenization']
    tokenizer_config = config['tokenizer']

    get_tokenizer_factory = utils.import_fn(tokenizer_config['module'])
    tokenizer_factory = get_tokenizer_factory(**tokenizer_config['module_kwargs'])
    tokenizer = tokenizer_factory()

    tokenize_hf_dataset.tokenize(**tokenization_config)
