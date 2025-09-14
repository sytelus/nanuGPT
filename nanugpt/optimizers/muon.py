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
              expect_embeddings: bool = True,
              expect_layers: bool = True,
              ):

    assigned = set()  # track parameters already assigned to a group (by id)

    # Head params (typically tied to token embeddings in GPT); ensure unique assignment
    head_module = getattr(model, head_name)
    head_weight = head_module.weight
    head_params = [head_weight]
    assigned.add(id(head_weight))

    # Embedding params: collect from all nn.Embedding modules; avoid duplicates and tied head weight
    embed_params = []
    embed_modules_found = 0
    for module in model.modules():
        if isinstance(module, nn.Embedding):
            embed_modules_found += 1
            for p in module.parameters(recurse=False):
                if p.ndim >= 1 and id(p) not in assigned:  # embeddings usually have a single weight tensor
                    embed_params.append(p)
                    assigned.add(id(p))

    # Hidden matrix params: parameters within Block modules having ndim >= 2
    hidden_matrix_params = []
    block_modules_found = 0
    for module in model.modules():
        if module.__class__.__name__ == layer_class_name:
            block_modules_found += 1
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

    # Expectations and validation
    if expect_embeddings and len(embed_params) == 0:
        raise RuntimeError(
            f"Muon optimizer expected to find embedding parameters (nn.Embedding), "
            f"but none were found. embed_modules_found={embed_modules_found}. "
            f"If your model uses a different embedding type or name, set expect_embeddings=False."
        )
    if expect_layers and len(hidden_matrix_params) == 0:
        raise RuntimeError(
            f"Muon optimizer expected to find transformer blocks with class name '{layer_class_name}' "
            f"containing matrix parameters (ndim>=2), but none were found. block_modules_found={block_modules_found}. "
            f"If your blocks have a different class name, pass layer_class_name accordingly or set expect_layers=False."
        )

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

    # Debug summaries for visibility
    def _numel(params):
        return int(sum(p.numel() for p in params))

    logging.summary({
        'muon/embedding_modules': embed_modules_found,
        'muon/block_modules': block_modules_found,
        'muon/head_params_tensors': len(head_params),
        'muon/embed_params_tensors': len(embed_params),
        'muon/scalar_params_tensors': len(scalar_params),
        'muon/hidden_matrix_params_tensors': len(hidden_matrix_params),
        'muon/head_params': _numel(head_params),
        'muon/embed_params': _numel(embed_params),
        'muon/scalar_params': _numel(scalar_params),
        'muon/hidden_matrix_params': _numel(hidden_matrix_params),
        'muon/param_groups_adam': len(adam_groups),
        'muon/param_groups_muon': 1 if hidden_matrix_params else 0,
        'muon/expect_embeddings': expect_embeddings,
        'muon/expect_layers': expect_layers,
    })

    optimizer = MuonWithAuxAdam(param_groups)
    return optimizer
