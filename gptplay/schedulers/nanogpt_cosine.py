import math
import numpy as np

from torch.optim import LRScheduler

from gptplay import logging

class NanoGptCosineScheduler(LRScheduler):
    def __init__(self, optimizer, warmup_iters: int, lr_decay_iters: int, min_lr: float,
                 last_epoch=-1, verbose=False):
        self.min_lr = min_lr
        self.warmup_iters = warmup_iters
        self.lr_decay_iters = lr_decay_iters  # after this LR stays constant
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            logging.warn("To get the last learning rate computed by the scheduler, please use `get_last_lr()`.")

        # get initial LR set in each group in optimizer
        init_lrs = np.fromiter(group['initial_lr'] \
                                for group in self.optimizer.param_groups)

         # 1) linear warmup for warmup_iters steps
        if self.last_epoch < self.warmup_iters:
            return (init_lrs * self.last_epoch / self.warmup_iters).tolist()

        # 2) if it > lr_decay_iters, return min learning rate
        if self.last_epoch > self.lr_decay_iters:
            return [self.min_lr] * len(init_lrs)

        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (self.last_epoch - self.warmup_iters) / (self.lr_decay_iters - self.warmup_iters)
        assert 0.0 <= decay_ratio <= 1.0
        # coeff ranges 0..1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return (self.min_lr + coeff * (init_lrs - self.min_lr)).tolist()


def get_scheduler(optimizer, warmup_iters: int, lr_decay_iters: int, min_lr: float):
    return NanoGptCosineScheduler(
        optimizer=optimizer, warmup_iters=warmup_iters, lr_decay_iters=lr_decay_iters,
        min_lr=min_lr)
