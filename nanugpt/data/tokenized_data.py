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
    """Wraps memmap array as a torch Dataset so that we can access sequence starting at any index"""
    def __init__(self, data:np.memmap, context_length:int):
        super().__init__()
        self.data = data
        self.context_length = context_length
        # we need minimum of 2 sequences to generate x and y
        assert len(data) >= context_length + 1, "dataset tokens must be at least context_length + 1, got %d" % len(data)
        # imagine moving a window of size context_length over data
        self.seq_count = len(data)-context_length+1

    def __len__(self):
        return self.seq_count

    def __getitem__(self, idx):
        # requrn sequence at idx
        return self.data[idx:idx+self.context_length]

class MemmapDataloader:
    # iterator to simulate dataloader
    def __init__(self, memmap_dataset:MemmapDataset, batch_size:int,
                 seed:int, shuffle:bool, wrap_around:bool):
        if shuffle and not wrap_around:
            raise ValueError("wrap_around must be True if shuffle is True")
        self.dataset = memmap_dataset
        self.wrap_around = wrap_around
        self.shuffle = shuffle

        # For N tokens, dataset size is S=N-context_len+1, i.e.,
        # we move window of size context_len over data but leave
        # For S consecutive sequences, we have S-1 pairs of consecutive x and y
        # So, total number of sequences is S-1
        self.n_seqs = len(self.dataset)-1

        # if batch size is greater than number of sequences, we will return trucated batch
        # unless wrap_around is True
        self.batch_size = min(batch_size, self.n_seqs) if not wrap_around else batch_size
        # how many batches will we return in one epochs
        self.batch_count = math.ceil(self.n_seqs/self.batch_size)
        # random generator for shuffling
        self.rand_gen = torch.Generator().manual_seed(seed)
        self.idx = 0 # index of batch

    def __iter__(self):
        return self

    def __next__(self):
        # If we returned same as count of batches then reset the iterator
        if self.idx >= len(self):
            self.idx = 0
            raise StopIteration

        if self.shuffle:
            # chose random indices for sequences that will be in batch
            ix = torch.randint(self.n_seqs, (self.batch_size,), generator=self.rand_gen)

            # wrap_around is ignored if shuffle is True
        else:
            # we are sequentially returning batches

            # adjust batch size if we don't have enough tokens left
            start_seq_index = self.idx * self.batch_size
            avail_seq_count = self.n_seqs - start_seq_index
            this_batch_size = min(self.batch_size, avail_seq_count)

            # first get indices for available sequences
            ix = torch.arange(start_seq_index, start_seq_index+this_batch_size)

            # if wrap around and remaining_seq_count > 0, then get remaining sequences from start
            if self.wrap_around:
                start_seq_index = 0
                remaining_batch_size = self.batch_size - this_batch_size
                while remaining_batch_size > 0:
                    avail_seq_count = self.n_seqs - start_seq_index
                    this_batch_size = min(remaining_batch_size, avail_seq_count)
                    ix = torch.cat((ix, torch.arange(start_seq_index, start_seq_index+this_batch_size)))
                    start_seq_index += this_batch_size
                    remaining_batch_size -= this_batch_size

        # sequence at index is x and sequence at next token is y
        x = torch.stack([torch.from_numpy((self.dataset[i]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((self.dataset[i+1]).astype(np.int64)) for i in ix])

        self.idx += 1

        return x, y

    def __len__(self):
        return self.batch_count

def get_data(local_rank:int, # everything except local rank comes from config
             context_length:int, dtype,
             device_batch_size:int, eval_batch_size:int,
             data_loader_seed:int,
             tokenized_train_path:str, tokenized_val_path:str,
             tokenized_test_path=None,
             shuffle=True,
             wrap_around=True):

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
                            seed=data_loader_seed+local_rank, shuffle=shuffle, wrap_around=wrap_around), \
            MemmapDataloader(val_dataset, eval_batch_size,
                             seed=data_loader_seed+local_rank, shuffle=shuffle, wrap_around=wrap_around), \
            MemmapDataloader(test_dataset, eval_batch_size,
                             seed=data_loader_seed+local_rank, shuffle=shuffle, wrap_around=wrap_around) if test_dataset else None
