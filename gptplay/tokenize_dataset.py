# saves the openwebtext dataset to a binary file for training. following was helpful:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py

from typing import Optional, Tuple, List, Dict, Mapping, Callable, MutableMapping
import math
import os
from multiprocessing import cpu_count, Pool
import numpy as np
from functools import partial

from tqdm.auto import tqdm

from datasets import DatasetDict, load_dataset, load_from_disk

import torch
from torch.utils.data import DataLoader

from gptplay.tokenizers.tokenizer_base import TokenizerBase
from gptplay import logging
from gptplay import utils
from gptplay.config import Config
from gptplay.data.hf_dataset import get_datasets


def tokenize(hf_name_path:str, hf_dataset_name:Optional[str], hf_data_dir:Optional[str], hf_data_files:Optional[str],
             train_split:Optional[str], val_split:Optional[str], test_split:Optional[str], hf_cache_dir:Optional[str],
             val_fraction:Optional[float], test_fraction:Optional[float],
             text_column:str, tokenizer_factory:Callable[[], TokenizerBase],
             tokenized_out_dir:str, data_loader_seed:int, hf_sample_by:Optional[str], hf_revision:Optional[str])->None:

    dataset = get_datasets(hf_name_path=hf_name_path, hf_dataset_name=hf_dataset_name, hf_data_dir=hf_data_dir, hf_data_files=hf_data_files,
                           train_split=train_split, val_split=val_split, test_split=test_split, hf_cache_dir=hf_cache_dir,
                           val_fraction=val_fraction, test_fraction=test_fraction, data_loader_seed=data_loader_seed,
                           hf_sample_by=hf_sample_by, hf_revision=hf_revision)

    class TokenizerPerThread:
        def __init__(self, tokenizer_factory):
            self._tok = None
            self.tokenizer_factory = tokenizer_factory

        def encode_text(self, text_or_row)->Mapping:
            if isinstance(text_or_row, Mapping): # could be LazyRow
                text = text_or_row[text_column if text_column else 'text']
            elif isinstance(text_or_row, str):
                text = text_or_row
            else:
                raise ValueError(f'encode_text expected str or Mapping, got {type(text_or_row)}')

            if self._tok is None:
                self._tok = tokenizer_factory()
            ids = self._tok.batch_encode([text])['input_ids'][0]
            ids.append(self._tok.eot_token_id())
            return {'ids': ids, 'len': len(ids)}

    # tokenize all splits in the dataset
    tokenized = dataset.map(
        partial(lambda tok, text: tok.encode_text(text), TokenizerPerThread(tokenizer_factory)),
        remove_columns=[text_column] if text_column else None,
        desc="tokenizing the splits",
        num_proc=utils.work_cpu_count(),
    )

    tok = tokenizer_factory()
    vocab_size = len(tok)
    logging.summary({'vocab_size': vocab_size})
    np_dtype = np.uint16 if vocab_size < 2**16 else np.uint32
    logging.summary({'np_dtype': str(np_dtype)})


    # concatenate all the ids in each dataset into one large file we can use for training
    for split in [train_split, val_split, test_split]:
        if split not in tokenized:
            continue
        dset = tokenized[split]

        arr_len = np.sum(dset['len'], dtype=np.uint64)
        logging.summary({f'{split}_tokens': arr_len})

        filename = os.path.join(utils.full_path(tokenized_out_dir, create=True) , f'{split}.bin')
        arr = np.memmap(filename, dtype=np_dtype, mode='w+', shape=(arr_len,))
        # each shard has 8192 samples, so we need to calculate how many shards we need
        total_batches = 2**math.ceil(math.log2((len(dset) // 8192) + 1))

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()

    logging.info(f'Tokenized dataset saved to {tokenized_out_dir}')


if __name__ == "__main__":
    # specify config file to use as first argument in commandline
    config = Config(default_config_filepath='configs/tokenize/tiktoken_gpt2.yaml')

    # setup output dirs and logging
    logging_config = config['logging']
    out_dir = config['tokenization']['tokenized_out_dir']
    if not logging_config['log_dir']:
        logging_config['log_dir'] = utils.full_path(out_dir, create=True)

    logger = logging.Logger(master_process=True,  **logging_config)

    tokenization_config = config['tokenization']
    tokenizer_config = config['tokenizer']

    get_tokenizer_factory = utils.import_fn(tokenizer_config['module'])
    tokenizer_factory = get_tokenizer_factory(**tokenizer_config['module_kwargs'])

    tokenize(tokenizer_factory=tokenizer_factory, **tokenization_config)

    logging.all_done()