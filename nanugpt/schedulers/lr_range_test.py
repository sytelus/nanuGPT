

import math
import numpy as np

from torch.optim.lr_scheduler import LRScheduler

from nanugpt import glogging as logging

class LRRangeTestScheduler(LRScheduler):
    def __init__(self, optimizer, max_steps:int, range_coeff:float=5.0,
                 last_epoch=-1):
        self.max_steps = max_steps
        self.range_coeff = range_coeff
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            logging.warn("To get the last learning rate computed by the scheduler, please use `get_last_lr()`.")

        # get initial LR set in each group in optimizer
        init_lrs = np.fromiter((group['initial_lr'] for group in self.optimizer.param_groups), dtype=np.float32)

        x = self.last_epoch / self.max_steps  # normalize to 0..1
        c = math.exp(self.range_coeff * (x-0.5))
        return (init_lrs * c).tolist()


def get_scheduler(optimizer, max_steps: int, range_coeff: float):
    return LRRangeTestScheduler(
        optimizer=optimizer, max_steps=max_steps, range_coeff=range_coeff)
