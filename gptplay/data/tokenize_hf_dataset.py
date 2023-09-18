# saves the openwebtext dataset to a binary file for training. following was helpful:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py

from typing import Optional, Tuple, List, Dict, Mapping, Callable
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


def tokenize(hf_name_path:str, hf_dataset_name:Optional[str], hf_data_dir:Optional[str], hf_data_files:Optional[str], hf_revision:Optional[str],
             train_split:Optional[str], val_split:Optional[str], test_split:Optional[str], hf_cache_dir:Optional[str],
             val_fraction:Optional[float], test_fraction:Optional[float],
             text_column:str, tokenizer_factory:Callable[[], TokenizerBase],
             tokenized_out_dir:str)->None:

    if os.path.isdir(hf_name_path):
        dataset = load_from_disk(hf_name_path)
    else:
        dataset = load_dataset(hf_name_path, dataset_name=hf_dataset_name, data_dir=hf_data_dir, data_files=hf_data_files, revision=hf_revision,
                               cache_dir=hf_cache_dir)

    # standardize splits
    if not isinstance(dataset, DatasetDict):
        dataset = DatasetDict({train_split: dataset})

    logging.info(f'Loaded dataset {hf_name_path}')
    for split in dataset.keys():
        logging.summary({f'{split}_rows': len(dataset[split])})

    # set default values
    train_split = train_split or 'train'
    val_split = val_split or 'validation'
    test_split = test_split or 'test'
    val_fraction = val_fraction or 0.
    test_fraction = test_fraction or 0.

    if test_fraction: # simplify code
        assert val_fraction > 0, 'test_fraction can only be used if val_fraction > 0'

    # create or get splits
    if val_split not in dataset and (val_fraction+test_fraction)>0.:
        splits = dataset[train_split].train_test_split(test_size=val_fraction+test_fraction, shuffle=True, seed=data_loader_seed)
        dataset[train_split] = splits['train']
        if val_fraction:
            if not test_fraction:
                # there is no test split
                dataset[val_split] = splits['test']
            else:
                # there is a test split
                splits = splits['test'].train_test_split(test_size=test_fraction/(val_fraction+test_fraction), shuffle=True, seed=data_loader_seed)
                dataset[val_split] = splits['train']
                dataset[test_split] = splits['test']
        else:
            # there is only a test split
            dataset[test_split] = splits['test']
    else:
        logging.info(f'Using existing val_split "{val_split}", ignoring val_fraction={val_fraction}')
        if test_split not in dataset and test_fraction:
            splits = dataset[train_split].train_test_split(test_size=test_fraction, shuffle=True, seed=data_loader_seed)
            dataset[train_split] = splits['train']
            dataset[test_split] = splits['test']
        else:
            logging.info(f'Using existing test_split "{test_split}", ignoring test_fraction={test_fraction}')


    class TokenizerPerThread:
        def __init__(self, tokenizer_factory):
            self._tok = None
            self.tokenizer_factory = tokenizer_factory

        def encode_text(self, text:str)->Mapping:
            if self._tok is None:
                self._tok = tokenizer_factory()
            ids = self._tok.batch_encode([text])['input_ids'][0]
            ids.append(self._tok.eot_token_id())
            return {'ids': ids, 'len': len(ids)}

    # tokenize all splits in the dataset
    tokenized = dataset.map(
        partial(lambda tok, text: tok.encode_text(text), TokenizerPerThread(tokenizer_factory)),
        remove_columns=[text_column],
        desc="tokenizing the splits",
        num_proc=max(1, cpu_count()//2),
    )

    tok = tokenizer_factory()
    vocab_size = len(tok)
    logging.summary({'vocab_size': vocab_size})
    np_dtype = np.uint16 if vocab_size < 2**16 else np.uint32

    # concatenate all the ids in each dataset into one large file we can use for training
    for split in [train_split, val_split, test_split]:
        if not split:
            continue
        dset = tokenized[split]

        arr_len = np.sum(dset['len'], dtype=np.uint64)
        logging.summary({f'{split}_tokens': arr_len})

        filename = utils.full_path(os.path.join(tokenized_out_dir, f'{split}.bin'))
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
