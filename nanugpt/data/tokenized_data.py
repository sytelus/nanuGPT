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
        self.seq_count = len(data)-context_length+1

    def __len__(self):
        return self.seq_count

    def __getitem__(self, idx):
        # requrn sequence at idx
        return self.data[idx:idx+self.context_length]

class MemmapDataloader:
    # iterator to simulate dataloader
    def __init__(self, memmap_dataset:MemmapDataset, batch_size:int,
                 seed:int, shuffle:bool):
        self.dataset = memmap_dataset
        # we need at least one sequence for y, so batch size is one less than dataset size
        self.n_seqs = len(self.dataset)-1
        self.batch_size = min(batch_size, self.n_seqs)
        self.batch_count = math.ceil(self.n_seqs/self.batch_size)
        self.rand_gen = torch.Generator().manual_seed(seed)
        self.idx = 0 # index of batch
        self.shuffle = shuffle

    def __iter__(self):
        return self

    def __next__(self):
        # Count the batchers returned so far and reset iterator
        if self.idx >= len(self):
            self.idx = 0
            raise StopIteration

        if self.shuffle:
            # chose random indices of batch size in dataset
            ix = torch.randint(self.n_seqs, (self.batch_size,), generator=self.rand_gen)
        else:
            seqs_seen = self.idx * self.batch_size
            remaining_seqs = self.n_seqs - seqs_seen
            this_batch_size = min(self.batch_size, remaining_seqs)

            # return batches starting at idx batches
            ix = torch.arange(seqs_seen, seqs_seen+this_batch_size)

        # sequence at index is x and next sequence is y
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
             tokenized_test_path=None):

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
                            seed=data_loader_seed+local_rank, shuffle=True), \
            MemmapDataloader(val_dataset, eval_batch_size,
                             seed=data_loader_seed+local_rank, shuffle=True), \
            MemmapDataloader(test_dataset, eval_batch_size,
                             seed=data_loader_seed+local_rank, shuffle=True) if test_dataset else None
