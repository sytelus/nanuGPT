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


