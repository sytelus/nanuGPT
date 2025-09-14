import inspect

import torch

from nanugpt.optimizers.muon_optim import MuonWithAuxAdam

def get_optim(model,
              layers_name='blocks', # name of attribute that has the transformer layers in the model
              embed_name_substr='embed', # substring of attribute names that are embeddings
              head_name='lm_head', # name of attribute that is the final head
              head_params_lr=0.22, # learning rate for the head
              embed_params_lr=0.6, # learning rate for the embeddings
              scalar_params_lr=0.04, # learning rate for the scalar params
              adam_betas=(0.8, 0.95), # betas for the Adam optimizer
              adam_eps=1e-10, # eps for the Adam optimizer
              hidden_matrix_params_lr=0.05, # learning rate for the hidden matrix params
              hidden_matrix_momentum=0.95, # momentum for the hidden matrix params
              ):

    # collect the parameters to optimize
    hidden_matrix_params = [p for n, p in getattr(model, layers_name).named_parameters() if p.ndim >= 2 and embed_name_substr not in n]
    embed_params = [p for n, p in model.named_parameters() if embed_name_substr in n]
    scalar_params = [p for p in model.parameters() if p.ndim < 2]
    head_params = [getattr(model, head_name).weight]

    # init the optimizer(s)
    from muon import MuonWithAuxAdam
    adam_groups = [dict(params=head_params, lr=head_params_lr), dict(params=embed_params, lr=embed_params_lr), dict(params=scalar_params, lr=scalar_params_lr)]
    adam_groups = [dict(**g, betas=adam_betas, eps=adam_eps, use_muon=False) for g in adam_groups]
    muon_group = dict(params=hidden_matrix_params, lr=hidden_matrix_params_lr, momentum=hidden_matrix_momentum, use_muon=True)
    param_groups = [*adam_groups, muon_group]
    optimizer = MuonWithAuxAdam(param_groups)

    return optimizer