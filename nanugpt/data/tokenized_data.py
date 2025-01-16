from typing import Optional, Tuple
import math
import numpy as np

import torch
from torch.utils.data import Dataset

from nanugpt import utils
from nanugpt import glogging as logging


"""
This module implements the `get_data` interface for tokenized data.
"""
class MemmapDataset(Dataset):
    """
    Wraps memmap or numpy array as a torch Dataset so that we can access sequence starting at any index.
    We always return output_len tokens, if necessary, by wrapping around the data.
    If output_len is not specified, it is assumed to be equal to context_length.
    """
    def __init__(self, data:np.ndarray, context_length:int, output_len:Optional[int]=None):
        super().__init__()
        self.data = data
        self.context_length = context_length
        # we need minimum of 2 sequences to generate x and y
        assert len(data) >= context_length, "dataset tokens must be at least context_length, got %d" % len(data)
        # imagine moving a window of size context_length over data
        self.seq_count = len(data)-context_length+1
        # how many tokens shall we return at a time is controlled by output_len
        self.set_output_len(output_len)

        assert self.seq_count >= 2, "dataset must have at least 2 sequences to generate x,y pairs, got %d" % self.seq_count

    def set_output_len(self, output_len:Optional[int]):
        """Additional function to change output_len to context_len+1 for training"""
        self.output_len = output_len if output_len else self.context_length
        assert self.output_len <= len(self.data), "output_len must be less than or equal to length of data"

    def token_count(self):
        return len(self.data)

    def __len__(self):
        return self.seq_count

    def __getitem__(self, idx:int):
        # requrn sequence at idx
        # if length of slice extends beyond end of data,
        # wrap around and concatenate from start and return the sequence
        if idx+self.output_len > len(self.data):
            # calculate how many tokens to wrap around and keep wraping around until we get output_len tokens
            tokens = self.data[idx:]
            remaining = idx+self.output_len-len(self.data)
            while remaining > len(self.data):
                tokens = np.concatenate((tokens, self.data))
                remaining -= len(self.data)
            if remaining > 0:
                tokens = np.concatenate((tokens, self.data[:remaining]))
            return tokens

        # return sequence of output_len tokens
        return self.data[idx:idx+self.output_len]

class MemmapDataloader:
    """
    DataLoader looks at the dataset as sequences of size context_length.
    It is simply iterator that returns batch_size number of sequences at each iteration.
    There are a few corner cases:
        1. What if number of sequences is less than batch_size?
            Wrap around and keep filling the batch. As we get more batches, we might get to
            uniform distribution of sequences.
        2. With shuffle off: What if we are near the end and cannot fill the batch?
            Wrap around and fill the batch. Don't return truncated batch.
        3. With shuffle on: Should we fill batch from continuous sequences? Or get random sequences?
            Getting random sequences is expensive. So, we should get continuous sequences.
    """
    def __init__(self, memmap_dataset:MemmapDataset, batch_size:int,
                 seed:int, shuffle:bool, start_seq_index:int=0):
        self.dataset = memmap_dataset

        # random generator for shuffling
        self.rand_gen = torch.Generator().manual_seed(seed)
        self.shuffle = shuffle
        self.n_seqs = len(self.dataset)

        self.batch_size = batch_size
        # how many batches will we return in one epochs (last batch may get wrapped around)
        self.batch_count = math.ceil(float(self.n_seqs)/self.batch_size/self.dataset.context_length)
        self.batch_index = 0

        assert start_seq_index < self.n_seqs-1, "start_seq_index must be 1 less than number of sequences"
        assert start_seq_index == 0 or not shuffle, "start_seq_index must be 0 if shuffle is on"
        self.idx = start_seq_index # index of sequence to start with

        # add 1 for shifted y sequence
        self.dataset.set_output_len(batch_size * self.dataset.context_length + 1)

    def __iter__(self):
        return self

    def __next__(self):
        # If we reached 2nd last sequence, wrap around
        if self.batch_index >= self.batch_count:
            self.batch_index = 0
            # we are not changing self.idx as we want to wrap around
            raise StopIteration

        if self.shuffle:
            # chose from 0..n_seqs-2
            start = int(torch.randint(self.n_seqs-1, (1,), generator=self.rand_gen).item())
        else:
            # we are sequentially returning batches
            start = self.idx

        tokens = self.dataset[start]

        # convert tokens to x, y sequences using tensor views
        # x is first batch_size*context_length tokens
        # y is next token
        x = torch.from_numpy(tokens[:-1].astype(np.int64)).view(self.batch_size, self.dataset.context_length)
        y = torch.from_numpy(tokens[1:].astype(np.int64)).view(self.batch_size, self.dataset.context_length)

        self.idx = (self.idx + x.numel()) % self.dataset.token_count()
        self.batch_index += 1

        return x, y

    def __len__(self):
        return self.batch_count

def get_split_info(data_len, world_size, global_rank, shuffle)->Tuple[int, int]:
    split_size = math.ceil(data_len / world_size)
    split_start = int((data_len-1) * float(global_rank) / world_size) \
                if not shuffle else 0
    return split_size, split_start

def get_allocations(total, workers):
    start, end, remaining = 0, 0, total
    allocations = []
    while workers > 0:
        if not remaining:
            return None
        this_alloc = math.floor(remaining / workers)
        end += this_alloc
        remaining -= this_alloc
        allocations.append((start, end))
        start = end
        workers -= 1
    return allocations

def load_tokenized_data(filepath:str, dtype, context_length:int,
                        world_size:int, global_rank:int,
                        shuffle:bool, use_memmap:bool):
    if use_memmap:
        data = np.memmap(filepath, dtype=dtype, mode='r')
    else:
        data = np.array(np.memmap(filepath, dtype=dtype, mode='r'))

    n_tokens = len(data) - 1    # subtract 1 for shifted y sequence
    # avaialble sequences
    n_context = math.floor(n_tokens / context_length)

    n_context_per_rank = n_context / world_size
    rank_start = int(n_context * float(global_rank) / world_size) * context_length
    rank_end = rank_start + math.ceil(n_context_per_rank) * context_length
    return data[rank_start:rank_end]

def get_data(context_length:int, dtype,
             device_batch_size:int, eval_batch_size:int,
             data_loader_seed:int,
             tokenized_train_path:str, tokenized_val_path:str,
             tokenized_test_path=None,
             shuffle=False,):

    world_size = utils.get_world_size()
    global_rank = utils.get_global_rank()
    local_world_size = utils.get_local_world_size()
    assert world_size > 0 and global_rank >= 0 and global_rank < world_size and local_world_size >= 1, f"Invalid values: Word size={world_size}, global rank={global_rank}, local world size={local_world_size}"

    train_file_size = val_file_size = test_file_size = 0
    if tokenized_train_path:
        tokenized_train_path = utils.full_path(tokenized_train_path)
        train_file_size = utils.file_size(tokenized_train_path)
    if tokenized_val_path:
        tokenized_val_path = utils.full_path(tokenized_val_path)
        val_file_size = utils.file_size(tokenized_val_path)
    if tokenized_test_path:
        tokenized_test_path = utils.full_path(tokenized_test_path)
        test_file_size = utils.file_size(tokenized_test_path)

    # get current RAM size
    ram_size = utils.ram_size()
    use_memmap = (train_file_size + val_file_size + test_file_size)*local_world_size > ram_size

    logging.summary({'data/train_file_size': train_file_size,
                    'data/val_file_size': val_file_size,
                    'data/test_file_size': test_file_size,
                    'data/ram_size': ram_size,
                    'data/local_world_size': local_world_size,
                    'data/use_memmap': use_memmap
                    })

    if use_memmap:
        train_dataset = MemmapDataset(np.memmap(tokenized_train_path, dtype=dtype, mode='r'),
                                    context_length)
        val_dataset = MemmapDataset(np.memmap(tokenized_val_path, dtype=dtype, mode='r'),
                                    context_length)
        test_dataset = MemmapDataset(np.memmap(tokenized_test_path, dtype=dtype, mode='r'),
                                    context_length) if tokenized_test_path else None
    else:
        train_dataset = MemmapDataset(np.array(np.memmap(tokenized_train_path, dtype=dtype, mode='r')),
                                    context_length)
        val_dataset = MemmapDataset(np.array(np.memmap(tokenized_val_path, dtype=dtype, mode='r')),
                                    context_length)
        test_dataset = MemmapDataset(np.array(np.memmap(tokenized_test_path, dtype=dtype, mode='r')),
                                    context_length) if tokenized_test_path else None


    train_split_size, train_split_start = get_split_info(len(train_dataset), world_size, global_rank, shuffle)
    val_split_size, val_split_start = get_split_info(len(val_dataset), world_size, global_rank, shuffle)
    test_split_size, test_split_start = get_split_info(len(test_dataset), world_size, global_rank, shuffle) if test_dataset else (0, 0)

    return MemmapDataloader(train_dataset,
                            device_batch_size,
                            start_seq_index=train_offset,
                            seed=data_loader_seed+global_rank, shuffle=shuffle), \
            MemmapDataloader(val_dataset, eval_batch_size,
                            start_seq_index=val_offset,
                            seed=data_loader_seed+global_rank, shuffle=shuffle), \
            MemmapDataloader(test_dataset, eval_batch_size,
                            start_seq_index=test_offset,
                            seed=data_loader_seed+global_rank, shuffle=shuffle) if test_dataset else None

