import torch

def get_optim(model_params, learning_rate, beta1, beta2, weight_decay, eps):
    return torch.optim.AdamW(
        model_params,
        lr=learning_rate,
        betas=(beta1, beta2),
        weight_decay=weight_decay,
        eps=eps)