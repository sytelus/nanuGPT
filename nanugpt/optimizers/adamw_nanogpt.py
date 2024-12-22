import inspect

import torch
from torch.distributed.optim import ZeroRedundancyOptimizer

from nanugpt import glogging as logging

def get_optim(model, learning_rate, weight_decay,
              beta1, beta2, eps, enable_fused, zero_stage):
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]

    # TODO: move this out of function call
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    logging.summary({'model/params_decay': num_decay_params,
                     'model/params_no_decay': num_nodecay_params})

    # Create AdamW optimizer and use the fused version if it is available
    use_fused = enable_fused and \
        'fused' in inspect.signature(torch.optim.AdamW).parameters
    extra_args = dict(fused=True) if use_fused else dict()

    # TODO: move this out of function call
    logging.summary({'model/use_fused_adamw': use_fused})

    if zero_stage:
        optim = ZeroRedundancyOptimizer(**optim_groups[0], # params_rref
                                        optimizer_class=torch.optim.AdamW,
                                        # optim args
                                        lr=learning_rate,
                                        betas=(beta1, beta2),
                                        eps=eps, **extra_args)
        optim.add_param_group(optim_groups[1])
        return optim

    return torch.optim.AdamW(
        optim_groups,
        lr=learning_rate,
        betas=(beta1, beta2),
        eps=eps, **extra_args)