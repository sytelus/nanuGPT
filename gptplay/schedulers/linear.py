import torch

def get_scheduler(optimizer, start_factor=1.e-8, total_iters=10):
    return torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor = start_factor,
        total_iters=total_iters
    )