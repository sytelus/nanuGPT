import numpy as np

from torch.utils.data import Dataset

class MemmapDataset(Dataset):
    def __init__(self, data, context_length):
        super().__init__()
        self.data = data
        self.context_length = context_length

    def __len__(self):
        return len(self.data) - self.context_length

    def __getitem__(self, idx):
        return self.data[idx:idx+self.context_length]

class MemmapDataloader:
    def __init__(self, memmap_dataset:MemmapDataset, batch_size:int,
                 seed:int, shuffle:bool):
        self.dataset = memmap_dataset
        self.batch_size = min(batch_size, len(self.dataset))
        self.rand_gen = torch.Generator().manual_seed(seed)
        self.idx = 0
        self.suffle = shuffle

    def __iter__(self):
        return self

    def __next__(self):
        if shuffle:
            ix = torch.randint(len(dataset), (batch_size,), generator=self.rand_gen)
        else:
            ix = torch.arange(self.idx, self.idx+self.batch_size)
            self.idx += self.batch_size
            if self.idx >= len(self.dataset):
                self.idx = 0
                raise StopIteration
        x = torch.stack([torch.from_numpy((data[i]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1]).astype(np.int64)) for i in ix])
        return x, y

    def __len__(self):
        return len(self.data) // self.batch_size

def get_data(context_length:int, dtype,
             train_batch_size:int, eval_batch_size:int,
             data_loader_seed:int, local_rank:int,
             tokenized_train_path:str, tokenized_val_path:str,
             tokenized_test_path=None):
    train_dataset = MemmapDataset(np.memmap(tokenized_train_path, dtype=dtype, mode='r'),
                                  context_length)
    val_dataset = MemmapDataset(np.memmap(tokenized_val_path, dtype=dtype, mode='r'),
                                  context_length)
    test_dataset = MemmapDataset(np.memmap(tokenized_test_path, dtype=dtype, mode='r'),
                                  context_length) if tokenized_test_path else None

    # shuffle on val and test is needed as we do sampling for evaluation
    return MemmapDataloader(train_dataset, train_batch_size,
                            seed=data_loader_seed+local_rank, shuffle=True), \
            MemmapDataloader(val_dataset, eval_batch_size,
                             seed=data_loader_seed+local_rank, shuffle=True), \
            MemmapDataloader(test_dataset, eval_batch_size,
                             seed=data_loader_seed+local_rank, shuffle=True) if test_dataset else None
