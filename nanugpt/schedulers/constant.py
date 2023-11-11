import math
import numpy as np

from torch.optim.lr_scheduler import LRScheduler

from nanugpt import glogging as logging

class ConstantWithCooldownScheduler(LRScheduler):
    def __init__(self, optimizer, const_lr:float, warmup_iters: int, num_steps: int, cooldown_iters: int,
                 last_epoch=-1, verbose=False):

        self.const_lr = const_lr

        self.warmup_iters = warmup_iters
        self.cooldown_iters = cooldown_iters
        self.num_steps = num_steps

        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step: # type: ignore
            logging.warn("To get the last learning rate computed by the scheduler, please use `get_last_lr()`.")

        # get initial LR set in each group in optimizer
        init_lrs = np.fromiter((group['initial_lr'] for group in self.optimizer.param_groups), dtype=np.float32)

         # 1) linear warmup for warmup_iters steps
        if self.last_epoch < self.warmup_iters:
            return (init_lrs * self.last_epoch / self.warmup_iters).tolist()

        # 2) if it > num_steps, return cooldown learning rate
        if self.last_epoch > (self.num_steps - self.cooldown_iters):
            cooldown_start = self.num_steps - self.last_epoch
            return (init_lrs * (cooldown_start / self.cooldown_iters)).tolist()

        # 3) in between, use constant learning rate
        return np.full_like(init_lrs, self.const_lr).tolist()


def get_scheduler(optimizer, const_lr:float, warmup_iters: int, num_steps: int, cooldown_iters: int):
    return ConstantWithCooldownScheduler(
        optimizer=optimizer,
            const_lr=const_lr,
            warmup_iters=warmup_iters,
            num_steps=num_steps,
            cooldown_iters=cooldown_iters
        )