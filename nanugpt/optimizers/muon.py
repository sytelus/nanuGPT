import torch.nn as nn
import torch.distributed as dist

from nanugpt.optimizers.muon_optim import MuonWithAuxAdam, SingleDeviceMuonWithAuxAdam
from nanugpt import glogging as logging

def get_optim(model,
              layer_class_name='Block',
              head_name='lm_head',
              head_params_lr=0.22,
              embed_params_lr=0.6,
              scalar_params_lr=0.04,
              adam_betas=(0.8, 0.95),
              adam_eps=1e-10,
              muon_lr=0.05,
              muon_momentum_min=0.85,
              muon_momentum_max=0.95,
              muon_momentum_warmup=300,  # number of muon steps to warmup momentum from min to max
              expect_embeddings: bool = True,
              expect_layers: bool = True,
              # accept all other args
              **kwargs
              ):

    # If wrapped with DistributedDataParallel, unwrap one level to the underlying module
    root_model = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model

    assigned = set()  # track parameters already assigned to a group (by id)

    # Head params (typically tied to token embeddings in GPT); ensure unique assignment
    head_module = getattr(root_model, head_name)
    head_weight = head_module.weight
    head_params = [head_weight]
    assigned.add(id(head_weight))

    # Embedding params: collect from all nn.Embedding modules; avoid duplicates and tied head weight
    embed_params = []
    embed_modules_found = 0
    for module in root_model.modules():
        if isinstance(module, nn.Embedding):
            embed_modules_found += 1
            for p in module.parameters(recurse=False):
                if p.ndim >= 1 and id(p) not in assigned:  # embeddings usually have a single weight tensor
                    embed_params.append(p)
                    assigned.add(id(p))

    # Hidden matrix params: parameters within Block modules having ndim >= 2
    hidden_matrix_params = []
    block_modules_found = 0
    for module in root_model.modules():
        if module.__class__.__name__ == layer_class_name:
            block_modules_found += 1
            for p in module.parameters(recurse=True):
                if p.ndim >= 2 and id(p) not in assigned:
                    hidden_matrix_params.append(p)
                    assigned.add(id(p))

    # Scalar params: any remaining parameters with ndim < 2
    scalar_params = []
    for p in root_model.parameters():
        if p.ndim < 2 and id(p) not in assigned:
            scalar_params.append(p)
            assigned.add(id(p))

    # Expectations and validation
    # Allow tied embeddings (e.g., embedding weight == head weight). In that case,
    # embed_params may be empty even though an nn.Embedding exists. Only require
    # that at least one embedding module is present when expect_embeddings=True.
    if expect_embeddings and embed_modules_found == 0:
        raise RuntimeError(
            f"Muon optimizer expected to find at least one nn.Embedding module, "
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

    # Decide DDP vs single-device early to shape muon group keys
    use_ddp = isinstance(model, nn.parallel.DistributedDataParallel) or (dist.is_available() and dist.is_initialized())

    param_groups = adam_groups
    if hidden_matrix_params:
        muon_group = dict(params=hidden_matrix_params,
                          lr=muon_lr,
                          momentum=muon_momentum_max,
                          min_momentum=muon_momentum_min,
                          max_momentum=muon_momentum_max,
                          momentum_warmup=muon_momentum_warmup,
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

    # Pick distributed or single-device variant based on DDP/is_initialized
    OptimCls = MuonWithAuxAdam if use_ddp else SingleDeviceMuonWithAuxAdam
    logging.info({'muon/optimizer_class': OptimCls.__name__, 'muon/use_ddp': use_ddp})
    logging.summary({'muon/optimizer_class': OptimCls.__name__, 'muon/use_ddp': use_ddp})
    optimizer = OptimCls(param_groups)
    return optimizer
