import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' # needed to avoid Jupyter kernal crash due to matplotlib
from typing import Callable, Tuple, Dict, List, Sequence
from itertools import groupby, chain
from collections import defaultdict
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
import math
import psutil
import random
import hashlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import json
from dataclasses import dataclass

import torch

# We setup env variable if debugging mode is detected for vs_code_debugging.
# The reason for this is that when Python multiprocessing is used, the new process
# spawned do not inherit 'pydevd' so those process do not get detected as in debugging mode
# even though they are. So we set env var which does get inherited by sub processes.
if 'pydevd' in sys.modules:
    os.environ['vs_code_debugging'] = 'True'
def is_debugging()->bool:
    return 'vs_code_debugging' in os.environ and os.environ['vs_code_debugging']=='True'

def full_path(path:str, create=False)->str:
    assert path
    path = os.path.realpath(
            os.path.expanduser(
                os.path.expandvars(path)))
    if create:
        os.makedirs(path, exist_ok=True)
    return path


def setup_sys(seed, max_threads=None):
    os.environ['NUMEXPR_MAX_THREADS'] = str(psutil.cpu_count(logical=False) // 2) if max_threads is None else str(max_threads)
    np.random.seed(seed)
    random.seed(seed)

def median(values):
    values = sorted(values)
    size = len(values)
    if size % 2 == 1:
        return values[int((size - 1) / 2)]
    return (values[int(size / 2 - 1)] + values[int(size / 2)]) / 2


class ExponentialMovingAverage:
    def __init__(self, weight=0.9, initial_value=0.):
        self.value: float = initial_value
        self.n: int = 0
        self.weight = weight
        self.last_good_value, self.last_good_n = self.value, self.n

    def add(self, x: float) -> float:
        if not math.isnan(self.value):
            self.last_good_value, self.last_good_n = self.value, self.n
        self.n += 1
        self.value = x * self.weight + self.last_good_value * (1 - self.weight)
        return self.value

class SmoothedDyDx:
    def __init__(self, y_ema_weight=0.8, x_ema_weight=0.8,
                 dy_ema_weight=0.9, dx_ema_weight=0.9,
                 dydx_ema_weight=0.95):


        self.value = 0.
        self.n = 0

        # smooth x and y
        self.y = ExponentialMovingAverage(y_ema_weight)
        self.x = ExponentialMovingAverage(x_ema_weight)

        # smooth deltas
        self.dy = ExponentialMovingAverage(dy_ema_weight)
        self.dx = ExponentialMovingAverage(dx_ema_weight)

        # smooth dy/dx
        self.dydx = ExponentialMovingAverage(dydx_ema_weight)


    def add(self, y: float, x: float) -> float:
        last_x, last_y = self.x.value, self.y.value

        self.y.add(y)
        self.x.add(x)

        dydx = 0.
        if self.x.n > 1:
            self.dy.add(self.y.value - last_y)
            self.dx.add(self.x.value - last_x)

            dydx = self.dydx.add(self.dy.value / self.dx.value)

        self.value = dydx
        self.n += 1

        return dydx

def save_list(l, filename):
    with open(filename, 'w') as f:
        for item in l:
            if isinstance(item, Sequence):
                for i in item:
                    f.write(f"{i}\t")
            else:
                f.write(f"{item}")
            f.write("\n")

def tensor_hash(tensor, sort=False):
    flattened = tensor.clone().detach().flatten()
    if sort:
        # Flatten and sort the tensor
        flattened, _ = torch.sort(flattened)

    # Convert to byte representation
    tensor_bytes = flattened.cpu().numpy().tobytes()

    # Compute the hash
    return hashlib.sha256(tensor_bytes).hexdigest()

def shuffle_tuple_of_lists(t:Tuple[List, ...])->Tuple[List, ...]:
    # Length of any member
    length = len(t[0])

    # Generate a permutation of indices
    permuted_indices = list(range(length))
    random.shuffle(permuted_indices)

    # Reorder each member of the tuple using the permuted indices
    shuffled = tuple([member[permuted_indices] for member in t])

    return shuffled

def save_dataloader(dl, filename: str):
    with open(filename, 'w') as f:
        for b in dl:
            inputs, labels = tuple(t for t in b)
            assert(len(inputs)==len(labels))
            for i,l in zip(inputs.tolist(), labels.tolist()):
                for num in i+[l]:
                    f.write(f"{num}\t")
                f.write("\n")

def load_json(doc):
    """Load json that could possibly be malformed"""
    try:
        return json.loads(doc)
    except:
        return None

def uhgroupby(iterable, key:Callable):
    """Group by key and return a dict of iterables"""
    return groupby(sorted(iterable, key=key), key=key)

def ugroupby(iterable, key:Callable, gather:Callable=lambda d,k,g: list(g)):
    d = {}
    for k, g in groupby(iterable, key=key):
        d[k] = gather(d, k, g)
    return d


def draw_histogram(data, xlabel='Values', ylabel='Frequency', title='Histogram', bins=None, log_x=False, log_y=False):
    unique_vals = np.unique(data)
    if len(unique_vals) < 10:
        bins = len(unique_vals)  # If the unique values are less than 10, make a bin for each
    elif bins is None:  # Automatic bin sizing using Freedman-Diaconis rule
        q75, q25 = np.percentile(data, [75 ,25])
        iqr = q75 - q25
        bin_width = 2 * iqr / (len(data) ** (1/3))
        bins = min(round((np.max(data) - np.min(data)) / bin_width), 1000)

    n, bins, patches = plt.hist(data, bins=bins, edgecolor='black')

    # Create a normalization object which scales data values to the range [0, 1]
    fracs = n / n.max()
    norm = mcolors.Normalize(fracs.min(), fracs.max())

    # Assigning a color for each bar using the 'viridis' colormap
    for thisfrac, thispatch in zip(fracs, patches):
        color = cm.viridis(norm(thisfrac))
        thispatch.set_facecolor(color)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    if log_x:
        plt.xscale('log')
    if log_y:
        plt.yscale('log')

    plt.show()

def setup_torch(seed:int,
    device_name:bool,
    dtype:str,
    enable_distributed:bool=False,
    print_precision:int=10,
    gradient_accumulation_steps_1gpu:int=1)->TorchSetupInfo:

    @dataclass
    class TorchSetupInfo:
        is_cuda:bool,
        is_distributed: bool
        rank: int
        local_rank: int
        world_size: int
        device_name:str
        is_master: bool
        seed_offset: int
        gradient_accumulation_steps: int

    # show Tensor shape first for tensor's rpresentation
    normal_repr = torch.Tensor.__repr__
    torch.Tensor.__repr__ = lambda self: f"{tuple(self.shape)}:{normal_repr(self)}" # type: ignore
    torch.set_printoptions(precision=print_precision)

    assert device_name == 'cuda' and torch.cuda.is_available(), 'cuda not available. Set device_name=cpu.'
    assert dtype != 'bfloat16' and (torch.cuda.is_available() and torch.cuda.is_bf16_supported()), 'bfloat16 not supported. Use float16 or float32.'
    assert enable_distributed and torch.distributed.is_available(), 'Distributed training not available. Set enable_distributed=False.'

    is_cuda = False
    is_distributed = False
    rank = 0
    local_rank = 0
    world_size = 1
    device_name = 'cpu'
    is_master = True
    seed_offset = 0
    gradient_accumulation_steps = gradient_accumulation_steps_1gpu

    if device_name == 'cuda' and torch.cuda.is_available():
        torch.backends.cudnn.enabled = enable_cuda
        torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
        is_cuda = True
        device_name = 'cuda'

    if enable_distributed and torch.distributed.is_available():
        if torch.distributed.is_initialized():
            is_distributed = True
            rank = torch.distributed.get_rank()
            local_rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
            is_master = rank == 0
            seed_offset = rank

            assert gradient_accumulation_steps % world_size == 0, f'gradient_accumulation_steps ({gradient_accumulation_steps}) must be divisible by ddp_world_size ({ddp_world_size})'
            gradient_accumulation_steps = gradient_accumulation_steps_1gpu // world_size

            if device_name == 'cuda':
                torch.cuda.set_device(local_rank)
                device_name = f'cuda:{local_rank}'

    if enable_cuda:
        torch.cuda.manual_seed(seed+seed_offset)
    torch.manual_seed(seed+seed_offset)

    return DistibutedInfo(is_cuda=is_cuda, is_distributed=is_distributed, rank=rank, local_rank=local_rank,
                          world_size=world_size, device_name=device_name, is_master=is_master,
                          seed_offset=seed_offset, gradient_accumulation_steps=gradient_accumulation_steps)

def save_checkpoint(out_dir:str, name:str, model:nn.Module, optimizer, scheduler,
                    step:int, best_val_loss:float):
    checkpoint = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'scheduler': scheduler.state_dict(),
                  'step': step,
                  'best_val_loss': best_val_loss}

    out_dir = full_path(out_dir, create=True)
    checkpoint_filepath = os.path.join(out_dir, f'{name}_{step}.pt')
    torch.save(checkpoint, os.path.join(out_dir, checkpoint_filepath))