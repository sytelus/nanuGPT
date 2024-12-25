import inspect

import torch

def get_optim(model, learning_rate, weight_decay,
              beta1, beta2, eps, enable_fused, zero_stage):
    assert zero_stage == 0, "ZeroRedundancyOptimizer is not supported with adamw.py"

    return torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(beta1, beta2),
        eps=eps,
        weight_decay=weight_decay,)