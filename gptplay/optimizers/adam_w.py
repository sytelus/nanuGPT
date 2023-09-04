import inspect

import torch

def get_optim(model_named_params, learning_rate, weight_decay,
              beta1, beta2, eps, enable_fused):
    return torch.optim.AdamW(
        [p for pn, p in model_named_params],
        lr=learning_rate,
        betas=(beta1, beta2),
        eps=eps,
        weight_decay=weight_decay,)