from typing import Optional
import numpy as np

from torch.optim.lr_scheduler import LRScheduler

from nanugpt import glogging as logging

class ConstantWithCooldownScheduler(LRScheduler):
    def __init__(self, optimizer, const_lr:float,
                 warmup_iters: int, max_iters: Optional[int], cooldown_iters: int,
                 end_factor: float,
                 last_epoch=-1):

        self.const_lr = const_lr

        self.warmup_iters = warmup_iters
        self.cooldown_iters = cooldown_iters
        self.end_factor = end_factor
        self.max_iters = max_iters

        assert (self.max_iters is None and self.cooldown_iters == 0) or (self.max_iters is not None and self.cooldown_iters >= 0), "max_iters must be specified if cooldown_iters > 0"
        assert self.max_iters is None or warmup_iters + cooldown_iters <= self.max_iters, f"warmup_iters + cooldown_iters {warmup_iters}+{cooldown_iters} must be less than max_iters {self.max_iters}"
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step: # type: ignore
            logging.warn("To get the last learning rate computed by the scheduler, please use `get_last_lr()`.")

        # get initial LR set in each group in optimizer
        init_lrs = np.fromiter((group['initial_lr'] for group in self.optimizer.param_groups), dtype=np.float32)

         # linear warmup for warmup_iters steps
        if self.last_epoch < self.warmup_iters:
            return (init_lrs * self.last_epoch / self.warmup_iters).tolist()

        if self.max_iters is not None and self.last_epoch >= self.max_iters: # return min LR
            return (init_lrs * self.end_factor).tolist()

        # if within cooldown, return linear decay down to min learning rate
        if self.max_iters is not None and self.last_epoch > (self.max_iters - self.cooldown_iters):
            min_lr = init_lrs * self.end_factor
            decay_ratio = (self.max_iters - self.last_epoch - 1) / self.cooldown_iters
            return ((init_lrs - min_lr) * decay_ratio + min_lr).tolist()

        # in between, use constant learning rate
        return np.full_like(init_lrs, self.const_lr).tolist()


def get_scheduler(optimizer, const_lr:float,
                  warmup_iters: int, max_iters: Optional[int], cooldown_iters: int,
                  end_factor:float):
    return ConstantWithCooldownScheduler(
            optimizer=optimizer,
            const_lr=const_lr,
            warmup_iters=warmup_iters,
            max_iters=max_iters,
            cooldown_iters=cooldown_iters,
            end_factor=end_factor,
        )