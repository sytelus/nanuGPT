import math
import numpy as np

from torch.optim.lr_scheduler import LRScheduler

from nanugpt import glogging as logging

class LinearScheduler(LRScheduler):
    def __init__(self, optimizer, warmup_iters: int, max_iters: int,
                 end_factor: float,
                 last_epoch=-1):
        self.end_factor = end_factor
        self.warmup_iters = warmup_iters
        self.max_iters = max_iters  # after this LR stays constant to value of init_lrs * end_factor

        assert warmup_iters <= max_iters, f"warmup_iters {warmup_iters} must be less than max_iters {max_iters}"

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step: # type: ignore
            logging.warn("To get the last learning rate computed by the scheduler, please use `get_last_lr()`.")

        # get initial LR set in each group in optimizer
        init_lrs = np.fromiter((group['initial_lr'] for group in self.optimizer.param_groups), dtype=np.float32)

         # linear warmup for warmup_iters steps
        if self.last_epoch < self.warmup_iters:
            return (init_lrs * self.last_epoch / self.warmup_iters).tolist()

        # if it > max_iters, return min learning rate
        if self.last_epoch >= self.max_iters:
            return (init_lrs * self.end_factor).tolist()

        # in between, use linear decay down to min learning rate
        min_lr = init_lrs * self.end_factor
        decay_ratio = (self.last_epoch - self.warmup_iters) / (self.max_iters - self.warmup_iters)
        #assert 0.0 <= decay_ratio <= 1.0
        coeff = 1.0 - decay_ratio
        #assert 0.0 <= coeff <= 1.0
        return ((init_lrs - min_lr)*coeff + min_lr).tolist()

def get_scheduler(optimizer, warmup_iters: int, max_iters: int, end_factor:float):
    return LinearScheduler(
        optimizer=optimizer, warmup_iters=warmup_iters, max_iters=max_iters,
        end_factor=end_factor)
