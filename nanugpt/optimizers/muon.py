import torch.nn as nn

from nanugpt.optimizers.muon_optim import MuonWithAuxAdam
from nanugpt import glogging as logging

def get_optim(model,
              layer_class_name='Block',
              head_name='lm_head',
              head_params_lr=0.22,
              embed_params_lr=0.6,
              scalar_params_lr=0.04,
              adam_betas=(0.8, 0.95),
              adam_eps=1e-10,
              hidden_matrix_params_lr=0.05,
              hidden_matrix_momentum=0.95,
              ):

    assigned = set()  # track parameters already assigned to a group (by id)

    # Head params (typically tied to token embeddings in GPT); ensure unique assignment
    head_module = getattr(model, head_name)
    head_weight = head_module.weight
    head_params = [head_weight]
    assigned.add(id(head_weight))

    # Embedding params: collect from all nn.Embedding modules; avoid duplicates and tied head weight
    embed_params = []
    for module in model.modules():
        if isinstance(module, nn.Embedding):
            for p in module.parameters(recurse=False):
                if p.ndim >= 1 and id(p) not in assigned:  # embeddings usually have a single weight tensor
                    embed_params.append(p)
                    assigned.add(id(p))

    # Hidden matrix params: parameters within Block modules having ndim >= 2
    hidden_matrix_params = []
    for module in model.modules():
        if module.__class__.__name__ == layer_class_name:
            for p in module.parameters(recurse=True):
                if p.ndim >= 2 and id(p) not in assigned:
                    hidden_matrix_params.append(p)
                    assigned.add(id(p))

    # Scalar params: any remaining parameters with ndim < 2
    scalar_params = []
    for p in model.parameters():
        if p.ndim < 2 and id(p) not in assigned:
            scalar_params.append(p)
            assigned.add(id(p))

    # Initialize the optimizer groups
    adam_groups = []
    if head_params:
        adam_groups.append(dict(params=head_params, lr=head_params_lr))
    if embed_params:
        adam_groups.append(dict(params=embed_params, lr=embed_params_lr))
    if scalar_params:
        adam_groups.append(dict(params=scalar_params, lr=scalar_params_lr))
    adam_groups = [dict(**g, betas=adam_betas, eps=adam_eps, use_muon=False) for g in adam_groups]

    param_groups = adam_groups
    if hidden_matrix_params:
        muon_group = dict(params=hidden_matrix_params,
                          lr=hidden_matrix_params_lr,
                          momentum=hidden_matrix_momentum,
                          use_muon=True)
        param_groups = [*param_groups, muon_group]

    optimizer = MuonWithAuxAdam(param_groups)
    return optimizer
