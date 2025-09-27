from typing import Optional
import numpy as np

from torch.optim.lr_scheduler import LRScheduler

from nanugpt import glogging as logging

class ConstantWithCooldownScheduler(LRScheduler):
    def __init__(self, optimizer,
                 warmup_iters: Optional[int], max_iters: Optional[int],
                 cooldown_iters: Optional[int], cooldown_frac: Optional[float],
                 end_factor: float,
                 last_epoch=-1):

        self.warmup_iters = warmup_iters
        self.end_factor = end_factor
        self.max_iters = max_iters
        self.cooldown_iters = cooldown_iters

        if cooldown_frac is not None:
            assert max_iters is not None, "max_iters must be specified if cooldown_frac is specified"
            assert cooldown_iters is None, "Only one of cooldown_iters or cooldown_frac should be specified"
            self.cooldown_iters = int(cooldown_frac * max_iters)

        if self.cooldown_iters is not None:
            assert self.max_iters is not None, "max_iters must be specified if cooldown_iters is specified"
            assert self.cooldown_iters < self.max_iters, "cooldown_iters must be less than max_iters"
            assert self.end_factor < 1.0, "end_factor must be less than 1.0 for cooldown to have an effect"

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step: # type: ignore
            logging.warn("To get the last learning rate computed by the scheduler, please use `get_last_lr()`.")

        # get initial LR set in each group in optimizer
        init_lrs = np.fromiter((group['initial_lr'] for group in self.optimizer.param_groups), dtype=np.float32)

        # if within cooldown, return linear decay down to min learning rate
        # prioritize cooldown over warmup if both are specified
        if self.cooldown_iters is not None and self.max_iters is not None and self.last_epoch > (self.max_iters - self.cooldown_iters):
            min_lr = init_lrs * self.end_factor
            decay_ratio = (self.max_iters - self.last_epoch - 1) / self.cooldown_iters
            return ((init_lrs - min_lr) * decay_ratio + min_lr).tolist()

        # linear warmup for warmup_iters steps
        if self.warmup_iters is not None and self.last_epoch < self.warmup_iters:
            return (init_lrs * self.last_epoch / self.warmup_iters).tolist()

        # if we ran past max_iters, return min LR
        if self.max_iters is not None and self.last_epoch >= self.max_iters:  # return min LR
            return (init_lrs * self.end_factor).tolist()

        # in between, use constant learning rate equal to initial lr
        return init_lrs.tolist()


def get_scheduler(optimizer,
                  warmup_iters: Optional[int]=None, max_iters: Optional[int]=None,
                  cooldown_iters: Optional[int]=None, cooldown_frac: Optional[float]=None,
                  end_factor: float=1.0):
    return ConstantWithCooldownScheduler(
            optimizer=optimizer,
            warmup_iters=warmup_iters,
            max_iters=max_iters,
            cooldown_iters=cooldown_iters,
            cooldown_frac=cooldown_frac,
            end_factor=end_factor,
        )
