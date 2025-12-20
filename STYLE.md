# nanuGPT Style Guide

This document outlines the coding style, conventions, and philosophy for the nanuGPT codebase. Follow these guidelines to maintain consistency across the project.

## Philosophy

nanuGPT is derived from Karpathy's nanoGPT but deliberately diverges in philosophy:

> **Original nanoGPT**: Obsession with minimal lines of code
> **nanuGPT**: Obsession with modularity and scalability

### Core Principles

1. **Modularity Over Brevity** - Clean separation of concerns trumps code golf
2. **Plugin-Based Flexibility** - Components are swappable via YAML config, not code changes
3. **Configuration-Driven** - YAML is the source of truth for experiment setup
4. **Explicit Over Implicit** - Clear parameter passing, type hints, distributed training awareness
5. **Reproducibility** - Detailed logging, config saving, seed management
6. **PyTorch Conventions** - Use standard Dataset/DataLoader/Scheduler interfaces
7. **Scalability-Ready** - Built-in distributed training support from day one

---

## Directory Structure

```
nanugpt/
├── nanugpt/                 # Main package
│   ├── models/              # Model implementations (one file per model family)
│   ├── data/                # Dataset/dataloader implementations
│   ├── optimizers/          # Optimizer implementations
│   ├── schedulers/          # LR scheduler implementations
│   ├── losses/              # Loss function implementations
│   ├── tokenizers/          # Tokenizer implementations
│   ├── scalers/             # Gradient scaler implementations
│   ├── config.py            # Configuration system
│   ├── common.py            # Setup and creation utilities
│   ├── train.py             # Main training loop
│   ├── utils.py             # Utility functions
│   └── glogging.py          # Logging system
├── configs/                 # YAML configuration files
│   ├── train_gpt2/          # GPT-2 training configs
│   ├── train_llama2/        # LLaMA training configs
│   └── tokenize/            # Tokenization configs
├── tests/                   # Unit tests
└── scripts/                 # Utility and cluster scripts
```

---

## Import Style

### Organization

Group imports in this order, separated by blank lines:

```python
# Standard library
import os
import math
import timeit
from typing import Mapping, Tuple, Optional, Callable, Union
from dataclasses import dataclass
from contextlib import nullcontext

# Third-party
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch import distributed as dist

# Local
from nanugpt import utils, common
from nanugpt import glogging as logging  # Rename to avoid conflicts
```

### Conventions

- **Prefer explicit imports** over wildcards (`from x import *`)
- **Alias local logging**: `from nanugpt import glogging as logging`
- **Use `from typing import`** for type hint types
- **Combine related imports**: `from torch import Tensor, nn`

---

## Naming Conventions

### Functions and Variables

```python
# Functions: snake_case with descriptive verbs
def get_model(...): ...
def setup_device(...): ...
def import_fn(...): ...
def create_checkpoint(...): ...

# Variables: snake_case, descriptive
train_loader = ...
global_batch_size = ...
grad_acc_steps = ...
```

### Classes

```python
# Classes: PascalCase
class GPTConfig: ...
class MemmapDataset: ...
class CausalSelfAttention: ...
class ModelWithLoss: ...
```

### Constants and Type Aliases

```python
# Type aliases: PascalCase
GetLossType = Callable[[Union[torch.Tensor, Mapping], torch.Tensor], Tuple[torch.Tensor, int]]
TokenizerFactory = Callable[[], Tokenizer]

# Module-level constants: UPPER_SNAKE_CASE (rare in this codebase)
DEFAULT_CONTEXT_LENGTH = 1024
```

### Factory Functions

All pluggable modules expose a `get_*` factory function:

```python
# Models
def get_model(vocab_size, get_loss, **kwargs) -> nn.Module: ...

# Optimizers
def get_optim(model, learning_rate, weight_decay, **kwargs) -> Optimizer: ...

# Schedulers
def get_scheduler(optimizer, warmup_iters, max_iters, **kwargs) -> LRScheduler: ...

# Data
def get_data(**kwargs) -> Tuple[DataLoader, DataLoader, DataLoader]: ...

# Losses
def get_loss_factory(**kwargs) -> GetLossType: ...
```

---

## Type Hints

### Required Everywhere

Use type hints for all function signatures:

```python
def import_fn(spec: str) -> Callable:
    """Import a function from a module."""
    module_name, fn_name = spec.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, fn_name)

def setup_device(config: Config, logger: Logger) -> Tuple[torch.device, AbstractContextManager, TorchInfo]:
    ...
```

### Common Type Patterns

```python
from typing import Mapping, MutableMapping, Tuple, Optional, Callable, Union, Any

# Optional for nullable parameters
def get_model(dropout: Optional[float] = None): ...

# Union for multiple types
def forward(input_ids: Union[torch.Tensor, Mapping]): ...

# Callable for function parameters
def train(get_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): ...

# Mapping for dict-like read-only access
def process_config(config: Mapping[str, Any]): ...

# MutableMapping when modification needed
def update_config(config: MutableMapping[str, Any]): ...
```

---

## Dataclasses

Use `@dataclass` for structured configuration and data containers:

```python
from dataclasses import dataclass

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True

@dataclass
class TorchInfo:
    is_accelerator: bool
    is_distributed: bool
    device_type: str
    dtype: str
    device_id: int
    device_name: str
    global_rank: int
    local_rank: int
    world_size: int
    is_master: bool
```

---

## Docstrings

### Module-Level

Every module should have a docstring explaining its purpose:

```python
"""
This module implements the `get_data` interface for tokenized data allowing
for fast and memory efficient data loading. The data is loaded in a memmap file
and accessed by a custom Dataset and DataLoader which have same interface as
PyTorch's Dataset and DataLoader.
"""
```

### Function-Level

Use docstrings for complex or public functions:

```python
def import_fn(spec: str) -> Callable:
    """Import a function from a module.

    The spec is in the form of 'module.submodule.function'.

    Args:
        spec: Fully qualified function path (e.g., 'nanugpt.models.nanogpt.get_model')

    Returns:
        The imported function object.

    Raises:
        ImportError: If the module cannot be imported.
        AttributeError: If the function doesn't exist in the module.
    """
```

### Inline Comments

Use inline comments sparingly, only when logic isn't self-evident:

```python
# Weight decay only on parameters with dim >= 2 (no bias or LayerNorm)
decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
```

---

## Plugin Architecture

### How It Works

Components are specified in YAML config with `module` + `module_kwargs`:

```yaml
model:
  module: 'nanugpt.models.nanogpt.get_model'
  module_kwargs:
    n_layer: 12
    n_embd: 768
    n_head: 12
```

Loaded dynamically at runtime:

```python
get_model = utils.import_fn(model_config['module'])
model = get_model(**model_config['module_kwargs'])
```

### Creating New Plugins

1. Create a new file in the appropriate directory (e.g., `nanugpt/models/my_model.py`)
2. Implement the factory function following the interface contract
3. Reference it in config: `module: 'nanugpt.models.my_model.get_model'`

Example optimizer plugin:

```python
# nanugpt/optimizers/my_optimizer.py
import torch

def get_optim(model, learning_rate: float, weight_decay: float,
              enable_fused: bool = False, zero_stage: int = 0, **kwargs):
    """Create optimizer following the standard interface.

    Args:
        model: The model to optimize
        learning_rate: Learning rate
        weight_decay: Weight decay coefficient
        enable_fused: Whether to use fused implementation
        zero_stage: ZeRO optimization stage (0 = disabled)

    Returns:
        Configured optimizer instance
    """
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}

    # Standard weight decay separation pattern
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]

    return torch.optim.AdamW(optim_groups, lr=learning_rate, fused=enable_fused)
```

---

## Error Handling

### Assertions for Preconditions

Use assertions to validate invariants and preconditions:

```python
assert self.seq_len <= len(self.data), "seq_len must be less than or equal to length of data"
assert torch_info.is_master == utils.is_master_process()
```

### Exceptions with Context

Provide detailed context in exception messages:

```python
raise RuntimeError(f'Path "{path}" could not be found in specified dictionary at "{part}"')

raise ValueError(f'train_split is None and no "train" split found in dataset {hf_name_path}')

# Preserve original exception context
except Exception as e:
    raise KeyError(
        f'The yaml key or command line argument "{key}" is likely not named correctly. '
        f'Original value: {original_val} (type: {original_type}). '
        f'Original exception: {e}'
    )
```

### Fail Fast

Check conditions early and fail with clear messages:

```python
if not os.path.exists(config_path):
    raise FileNotFoundError(f"Config file not found: {config_path}")
```

---

## PyTorch Patterns

### Dataset/DataLoader

Follow PyTorch conventions:

```python
class MemmapDataset(Dataset):
    def __init__(self, data: np.ndarray, context_length: int):
        super().__init__()
        self.data = data
        self.context_length = context_length
        self.seq_count = len(data) - context_length + 1

    def __len__(self) -> int:
        return self.seq_count

    def __getitem__(self, idx: int) -> np.ndarray:
        return self.data[idx:idx + self.context_length]
```

### Model Definition

```python
class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        ...
```

### Distributed Training

Be explicit about distributed training:

```python
# Check if distributed
if torch_info.is_distributed:
    model = DistributedDataParallel(
        model,
        device_ids=[torch_info.device_id],
        gradient_as_bucket_view=True
    )

# Sync metrics across ranks
if torch_info.is_distributed:
    dist.reduce(metric_tensor, dst=0, op=dist.ReduceOp.SUM)
    dist.barrier()
```

---

## Configuration System

### YAML Structure

Organize configs by logical sections:

```yaml
general:      # Project-wide settings
  project_name: 'my_experiment'
  out_dir: '$OUT_DIR/runs'
  device_type: 'cuda'
  seed: 42
  dtype: 'bfloat16'

model:        # Model architecture
  module: 'nanugpt.models.nanogpt.get_model'
  module_kwargs:
    n_layer: 12
    n_embd: 768

data:         # Data loading
  module: 'nanugpt.data.tokenized_data.get_data'
  module_kwargs:
    batch_size: 64

training:     # Training dynamics
  max_steps: 10000
  grad_clip: 1.0

optimizer:    # Optimizer settings
  module: 'nanugpt.optimizers.adamw.get_optim'
  module_kwargs:
    learning_rate: 3.0e-4
```

### Config Features

```yaml
# Inheritance
__include__: 'base_config.yaml'

# Value copying (reference other config values)
scheduler:
  module_kwargs:
    max_iters: '_copy: /training/max_steps'

# Environment variables
general:
  out_dir: '$OUT_DIR/experiments'

# Timestamps
general:
  run_name: 'exp_{_time:%Y%m%d-%H%M%S}'

# Disable inheritance for a section
model:
  _inherit: false
  module: 'nanugpt.models.custom.get_model'
```

---

## Testing

### Test Organization

- One test file per module: `tests/test_memmap_dataloader.py`
- Use `unittest.TestCase` as the base class
- Group related tests in the same class

### Test Style

```python
import unittest
from unittest.mock import MagicMock

class TestMemmapDataloader(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.mock_dataset = MagicMock(spec=MemmapDataset)
        self.mock_dataset.context_length = 3
        self.mock_dataset.__len__.return_value = 10

    def test_batch_size_greater_than_dataset_length(self):
        """Batch size larger than dataset should produce single batch."""
        dataloader = MemmapDataloader(
            self.mock_dataset,
            batch_size=10,
            seed=42,
            shuffle=False
        )
        self.assertEqual(dataloader.batch_count, 1)
        batches = list(dataloader)
        self.assertEqual(len(batches), 1)

    def test_shuffle_produces_different_order(self):
        """Shuffling should change iteration order."""
        ...
```

### What to Test

- Edge cases (empty inputs, boundary conditions)
- Core functionality of plugins
- Configuration parsing and inheritance
- Data loading and batching logic

---

## Logging

### Use the Custom Logger

```python
from nanugpt import glogging as logging

logger = logging.Logger(config=logging_config, is_master=torch_info.is_master)

# Levels
logger.debug("Detailed info for debugging")
logger.info("Normal operational messages")
logger.warn("Warning conditions")
logger.error("Error conditions")
```

### Metric Logging

```python
# Log summary metrics (shown once)
logger.summary({
    "run/grad_acc_steps": grad_acc_steps,
    "run/global_batch_size": global_batch_size,
    "model/params_all": total_params,
})

# Log step metrics (shown each step)
logger.info({
    "train/step": step,
    "train/loss": loss.item(),
    "train/lr": current_lr,
})
```

### Artifact Logging

```python
logger.log_artifact(
    name='config',
    type='yaml',
    file_or_dir=config_filepath,
    desc_markdown="Configuration file at the start of the run"
)
```

---

## Code Quality

### Avoid Over-Engineering

- Don't add abstractions for single-use code
- Don't add configuration for things that won't change
- Don't add error handling for impossible cases
- Three similar lines is better than a premature abstraction

### Keep It Simple

```python
# Good: Direct and clear
optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wd)

# Avoid: Unnecessary abstraction
optimizer = OptimizerFactory.create('adamw', params, config=opt_config)
```

### Delete Dead Code

Don't comment out code or add `# removed` markers. Delete unused code entirely.

---

## Checklist for New Code

Before submitting code, verify:

- [ ] Type hints on all function signatures
- [ ] Docstring on public/complex functions
- [ ] Follows plugin interface if adding new component
- [ ] Uses existing utilities from `utils.py`
- [ ] Handles distributed training correctly
- [ ] Includes tests for core functionality
- [ ] No hardcoded paths (use config or env vars)
- [ ] Clear, descriptive variable names
- [ ] Assertions for preconditions
- [ ] Detailed exception messages
