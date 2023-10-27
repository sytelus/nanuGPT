# Run it with:
# torchrun --standalone --nproc_per_node=2 scripts/oneoff_tests/dist_test.py

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

    t = torch.tensor([1,3], device='cuda:{}'.format(dist.get_rank()))
    print("Rank {} has {}".format(dist.get_rank(), t))

    dist.barrier()

    dist.reduce(t, dst=0, op=dist.ReduceOp.SUM) # only rank 0 will get reduction result

    dist.barrier()

    print("Rank {} has {}".format(dist.get_rank(), t))

    dist.destroy_process_group()

