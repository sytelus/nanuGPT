# torchrun --standalone --nproc_per_node=8 scripts/play.py
import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributed as dist

if __name__ == "__main__":
    dist.init_process_group(backend='nccl', init_method='env://')
    print("Hello from rank {}".format(dist.get_rank()))

    t = torch.ones(1, device='cuda:{}'.format(dist.get_rank()))
    print("Rank {} has {}".format(dist.get_rank(), t))

    #dist.barrier()

    dist.reduce(t, 0)

    print("Rank {} has {}".format(dist.get_rank(), t))

    dist.destroy_process_group()

