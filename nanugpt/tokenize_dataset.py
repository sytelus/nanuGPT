# saves the openwebtext dataset to a binary file for training. following was helpful:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py

from typing import Optional, Mapping, Callable
import math
import os
import numpy as np
from functools import partial

from tqdm.auto import tqdm

from nanugpt import glogging as logging
from nanugpt import common
from nanugpt.tokenizers.tokenizer_base import TokenizerBase
from nanugpt import utils
from nanugpt.data.hf_dataset import get_datasets

"""
Tokenizes HuggingFace datasets.
"""

def tokenize(hf_name_path:str, hf_dataset_name:Optional[str], hf_data_dir:Optional[str], hf_data_files:Optional[str],
             train_split:Optional[str], val_split:Optional[str], test_split:Optional[str], hf_cache_dir:Optional[str],
             val_fraction:Optional[float], test_fraction:Optional[float],
             text_column:str, tokenizer_factory:Callable[[], TokenizerBase],
             tokenized_out_dir:str, data_loader_seed:int, hf_sample_by:Optional[str], hf_revision:Optional[str])->None:

    """
    This function uses same params as get_datasets in hf_dataset.py to load the HF dataset which may be on HF hub or local or bunch of files in folder on disk.

    Two additional params are:
    - text_column: str: name of the column in the dataset that contains the text to tokenize
    - tokenizer_factory: Callable[[], TokenizerBase]: a function that returns an instance of the tokenizer to use

    """


    common.check_env_vars()

    dataset, train_split, val_split, test_split = get_datasets(hf_name_path=hf_name_path, hf_dataset_name=hf_dataset_name, hf_data_dir=hf_data_dir, hf_data_files=hf_data_files,
                           train_split=train_split, val_split=val_split, test_split=test_split, hf_cache_dir=hf_cache_dir,
                           val_fraction=val_fraction, test_fraction=test_fraction, data_loader_seed=data_loader_seed,
                           hf_sample_by=hf_sample_by, hf_revision=hf_revision)

    # create instance so any downloading can be done before we start the tokenization
    tok = tokenizer_factory()
    vocab_size = len(tok)
    logging.summary({'run/vocab_size': vocab_size})
    np_dtype = np.uint16 if vocab_size < 2**16 else np.uint32
    logging.summary({'run/np_dtype': str(np_dtype)})

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
            ids.append(self._tok.eot_token_id()) # Always append EOT at end so two docs are separated
            return {'ids': ids, 'len': len(ids)}

    # tokenize all splits in the dataset
    tokenized = dataset.map(
        partial(lambda tok, text: tok.encode_text(text), TokenizerPerThread(tokenizer_factory)),
        remove_columns=[text_column] if text_column else None,
        desc="tokenizing the splits",
        num_proc=utils.work_cpu_count(),
    )

    # concatenate all the ids in each dataset into one large file we can use for training
    for split in [train_split, val_split, test_split]:
        if split not in tokenized:
            continue
        dset = tokenized[split]

        arr_len = np.sum(dset['len'], dtype=np.uint64)
        logging.summary({f'data/{split}_tokens': arr_len.item()})

        tokenized_out_dir = utils.full_path(tokenized_out_dir, create=True)
        filename = os.path.join(tokenized_out_dir, f'{split}.bin')
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
        del arr

    logging.info(f'Tokenized dataset saved to {tokenized_out_dir}')
