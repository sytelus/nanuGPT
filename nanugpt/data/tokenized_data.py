import math
import numpy as np

import torch
from torch.utils.data import Dataset

from nanugpt import utils

"""
This module implements the `get_data` interface for tokenized data allowing
for fast and memory efficient data loading. The data is loaded in a memmap file
and accessed by a custom Dataset and DataLoader which have same interface as
PyTorch's Dataset and DataLoader.
"""
class MemmapDataset(Dataset):
    """
    Wraps memmap array as a torch Dataset so that we can access sequence starting at any index.
    The dataset is still accessed token by token by specifying index but we always get
    context_length tokens starting at index.
    """
    def __init__(self, data:np.memmap, context_length:int):
        super().__init__()
        self.data = data
        self.context_length = context_length
        # we need minimum of 2 sequences to generate x and y
        assert len(data) >= context_length, "dataset tokens must be at least context_length, got %d" % len(data)
        # imagine moving a window of size context_length over data
        self.seq_count = len(data)-context_length+1
        assert self.seq_count >= 2, "dataset must have at least 2 sequences to generate x,y pairs, got %d" % self.seq_count

    def __len__(self):
        return self.seq_count

    def __getitem__(self, idx):
        # requrn sequence at idx
        return self.data[idx:idx+self.context_length]

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
        # wrap around allows to train on small dataset with small context length but
        # using large batch size
        self.shuffle = shuffle

        self.n_seqs = len(self.dataset)

        self.batch_size = batch_size
        # how many batches will we return in one epochs (last batch may get wrapped around)
        self.batch_count = math.ceil(self.n_seqs/self.batch_size)
        # random generator for shuffling
        self.rand_gen = torch.Generator().manual_seed(seed)

        assert start_seq_index < self.n_seqs-1, "start_seq_index must be 1 less than number of sequences"
        assert start_seq_index == 0 or not shuffle, "start_seq_index must be 0 if shuffle is on"
        self.idx = start_seq_index # index of sequence to start with

    def __iter__(self):
        return self

    def __next__(self):
        # If we reached 2nd last sequence, wrap around
        if self.idx >= self.n_seqs-1:
            self.idx = 0
            raise StopIteration

        if self.shuffle:
            # chose from 0..n_seqs-2, wrap around by n_seqs-1 (so that we have 1 seq left at end)
            # start_seq_index is not used in shuffle mode
            ix = (torch.randint(self.n_seqs-1, (1,), generator=self.rand_gen) \
                    + torch.arange(self.batch_size)) % (self.n_seqs-1)
        else:
            # we are sequentially returning batches
            ix = (self.idx + torch.arange(self.batch_size)) % (self.n_seqs-1)

        self.idx += len(ix)

        # sequence at index is x and sequence at next token is y
        x = torch.stack([torch.from_numpy((self.dataset[i]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((self.dataset[i+1]).astype(np.int64)) for i in ix])

        return x, y

    def __len__(self):
        return self.batch_count

def get_data(global_rank:int, # everything except global_rank comes from config
             context_length:int, dtype,
             device_batch_size:int, eval_batch_size:int,
             data_loader_seed:int,
             tokenized_train_path:str, tokenized_val_path:str,
             tokenized_test_path=None,
             shuffle=True,):

    if tokenized_train_path:
        tokenized_train_path = utils.full_path(tokenized_train_path)
    if tokenized_val_path:
        tokenized_val_path = utils.full_path(tokenized_val_path)
    if tokenized_test_path:
        tokenized_test_path = utils.full_path(tokenized_test_path)
    train_dataset = MemmapDataset(np.memmap(tokenized_train_path, dtype=dtype, mode='r'),
                                  context_length)
    val_dataset = MemmapDataset(np.memmap(tokenized_val_path, dtype=dtype, mode='r'),
                                  context_length)
    test_dataset = MemmapDataset(np.memmap(tokenized_test_path, dtype=dtype, mode='r'),
                                  context_length) if tokenized_test_path else None

    # shuffle on val and test is needed as we do sampling for evaluation
    return MemmapDataloader(train_dataset, device_batch_size,
                            seed=data_loader_seed+global_rank, shuffle=shuffle), \
            MemmapDataloader(val_dataset, eval_batch_size,
                             seed=data_loader_seed+global_rank, shuffle=shuffle), \
            MemmapDataloader(test_dataset, eval_batch_size,
                             seed=data_loader_seed+global_rank, shuffle=shuffle) if test_dataset else None
