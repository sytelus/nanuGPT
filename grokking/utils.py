from typing import Tuple, Dict, List
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
import math
import psutil
import random

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

def setup_torch():
    torch.backends.cudnn.enabled = True
    torch.set_printoptions(precision=10)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    os.environ['NUMEXPR_MAX_THREADS'] = str(psutil.cpu_count(logical=False) // 2)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)