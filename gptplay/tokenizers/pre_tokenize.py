# saves the openwebtext dataset to a binary file for training. following was helpful:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py

import math
import os
from multiprocessing import cpu_count, Pool
import numpy as np
from functools import partial

import tiktoken
from datasets import load_dataset # huggingface datasets
from tqdm.auto import tqdm

from gptplay import utils


def prepare(dataset_path:str, tokenized_out_dir:str,
            dataset_name:str, data_files=None, dataset_save_dir=None):

    dataset = load_dataset(dataset_path, name=dataset_name, data_files=data_files)

    print(f'Loaded dataset {dataset_name}')
    for split in dataset.keys():
        print(f'Split {split} has {len(dataset[split])} rows')

    if dataset_save_dir:
        dataset_save_dir = utils.full_path(dataset_save_dir, create=True)
        dataset.save_to_disk(dataset_save_dir)
        print(f'Saved dataset to {dataset_save_dir}')

    class TikTokenFactory:
        def __init__(self):
            self._enc = None
            self.eot_token = None

        def encode_ordinary(self, text):
            if self._enc is None:
                self._enc = tiktoken.get_encoding("gpt2")
                self.eot_token = self._enc.eot_token
            return self._enc.encode_ordinary(text)

    # we now want to tokenize the dataset. first define the encoding function (gpt2 bpe)
    def process(enc, example):
        ids = enc.encode_ordinary(example['text']) # encode_ordinary ignores any special tokens
        ids.append(enc.eot_token) # add the end of text token, e.g. 50256 for gpt2 bpe
        # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
        out = {'ids': ids, 'len': len(ids)}
        return out

    # tokenize the dataset
    tokenized = dataset.map(
        partial(process, TikTokenFactory()),
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=max(1, cpu_count()//2),
    )

    tokenized_out_dir = utils.full_path(os.path.join(tokenized_out_dir, dataset_name, 'tiktoken'), create=True)

    # concatenate all the ids in each dataset into one large file we can use for training
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        print(f'{split} has {arr_len} tokens')
        filename = os.path.join(tokenized_out_dir, f'{split}.bin')
        dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
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

    print(f'Tokenized dataset saved to {tokenized_out_dir}')
