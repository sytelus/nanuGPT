import inspect

import torch

def get_optim(model, learning_rate, weight_decay,
              beta1, beta2, eps, enable_fused):
    return torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(beta1, beta2))