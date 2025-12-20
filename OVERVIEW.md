# nanuGPT Technical Overview

This document provides a comprehensive technical overview of the nanuGPT codebase for maintainers and contributors.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Configuration System](#configuration-system)
3. [Plugin Architecture](#plugin-architecture)
4. [Training Pipeline](#training-pipeline)
5. [Data Pipeline](#data-pipeline)
6. [Distributed Training](#distributed-training)
7. [Mixed Precision & Gradient Accumulation](#mixed-precision--gradient-accumulation)
8. [Checkpoint System](#checkpoint-system)
9. [Logging System](#logging-system)
10. [Generation & Inference](#generation--inference)
11. [Key Files Reference](#key-files-reference)

---

## Architecture Overview

nanuGPT follows a **plugin-based modular architecture** where components are decoupled and swappable via YAML configuration.

```
┌─────────────────────────────────────────────────────────────────┐
│                        train.py / generate.py                   │
├─────────────────────────────────────────────────────────────────┤
│                         common.py (setup)                       │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │
│  │  Models  │ │   Data   │ │Optimizers│ │Schedulers│  Plugins  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘           │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐                        │
│  │  Losses  │ │Tokenizers│ │ Scalers  │                        │
│  └──────────┘ └──────────┘ └──────────┘                        │
├─────────────────────────────────────────────────────────────────┤
│  config.py │ utils.py │ glogging.py │ other utilities          │
└─────────────────────────────────────────────────────────────────┘
```

### Core Components

| Component | Location | Purpose |
|-----------|----------|---------|
| `train.py` | Root + `nanugpt/train.py` | Entry point and training loop |
| `generate.py` | Root + `nanugpt/generate.py` | Text generation/inference |
| `config.py` | `nanugpt/config.py` | YAML configuration with inheritance |
| `common.py` | `nanugpt/common.py` | Setup utilities and plugin loading |
| `utils.py` | `nanugpt/utils.py` | General utilities and distributed helpers |
| `glogging.py` | `nanugpt/glogging.py` | Logging with W&B integration |

---

## Configuration System

The configuration system (`nanugpt/config.py`) provides powerful YAML-based configuration with several advanced features.

### Config Hierarchy

```
configs/base_config.yaml          # Global defaults
    ↓ (inherited by)
configs/train_gpt2/base.yaml      # Model family defaults
    ↓ (inherited by)
configs/train_gpt2/experiment.yaml # Experiment-specific
    ↓ (overridden by)
Command-line arguments            # Runtime overrides
```

### Key Features

#### 1. Inheritance (`__include__`)

```yaml
# experiment.yaml
__include__: 'base_config.yaml'

model:
  module_kwargs:
    n_layer: 24  # Override base config
```

#### 2. Value Copying (`_copy:`)

Reference values from other config sections:

```yaml
scheduler:
  module_kwargs:
    warmup_iters: 100
    max_iters: '_copy: /training/max_steps'  # Copies from training.max_steps
```

#### 3. Environment Variables

```yaml
general:
  out_dir: '$OUT_DIR/experiments'
  data_root: '$DATA_ROOT/datasets'
```

#### 4. Timestamps (`_time:`)

```yaml
general:
  run_name: 'exp_{_time:%Y%m%d-%H%M%S}'
```

#### 5. Command-Line Overrides

```bash
python train.py config.yaml --training.max_steps 50000 --model.module_kwargs.n_layer 12
```

Type preservation: string arguments are converted to match the original config type.

#### 6. Disable Inheritance (`_inherit: false`)

```yaml
model:
  _inherit: false  # Don't inherit model section from base
  module: 'nanugpt.models.custom.get_model'
```

### Config Sections

| Section | Purpose |
|---------|---------|
| `general` | Project name, output dir, device type, seed, dtype |
| `model` | Model architecture plugin and hyperparameters |
| `data` | Data loading plugin and batch sizes |
| `training` | Training dynamics: max_steps, grad_clip, batch sizes |
| `optimizer` | Optimizer plugin and learning rate settings |
| `scheduler` | LR scheduler plugin and warmup settings |
| `loss` | Loss function plugin |
| `scaler` | Gradient scaler plugin (for mixed precision) |
| `eval` | Evaluation frequency, checkpoint settings |
| `logging` | Console/file/W&B logging configuration |
| `tokenizer` | Tokenizer plugin for generation |

---

## Plugin Architecture

Components are loaded dynamically at runtime using the `module` + `module_kwargs` pattern.

### How It Works

```yaml
# In config
optimizer:
  module: 'nanugpt.optimizers.adamw_nanogpt.get_optim'
  module_kwargs:
    learning_rate: 3.0e-4
    weight_decay: 0.1
```

```python
# In code (common.py / train.py)
get_optim = utils.import_fn(optimizer_config['module'])
optimizer = get_optim(model, **optimizer_config['module_kwargs'])
```

### Plugin Interfaces

Each plugin type follows a standard interface:

#### Models
```python
def get_model(vocab_size: int, get_loss: Callable, **kwargs) -> nn.Module:
    """Returns model wrapped with ModelWithLoss."""
```

#### Data
```python
def get_data(**kwargs) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """Returns (train_loader, val_loader, test_loader)."""
```

#### Optimizers
```python
def get_optim(model: nn.Module, learning_rate: float, weight_decay: float,
              enable_fused: bool, zero_stage: int, **kwargs) -> Optimizer:
    """Returns configured optimizer."""
```

#### Schedulers
```python
def get_scheduler(optimizer: Optimizer, warmup_iters: int,
                  max_iters: int, **kwargs) -> LRScheduler:
    """Returns learning rate scheduler."""
```

#### Losses
```python
def get_loss_factory(**kwargs) -> Callable[[Tensor, Tensor], Tuple[Tensor, int]]:
    """Returns loss function that computes (loss, correct_count)."""
```

### Available Plugins

| Type | Plugins |
|------|---------|
| Models | `nanogpt`, `hf_gpt2`, `hf_llama`, `tinyllama`, `te_llama3` |
| Data | `tokenized_data`, `hf_dataset`, `grokking_data`, `arithmatic_data` |
| Optimizers | `adamw`, `adamw_nanogpt`, `muon` |
| Schedulers | `cosine`, `linear`, `constant`, `lr_range_test` |
| Losses | `autoregressive_loss`, `grokking_loss`, `arithmatic_loss` |
| Tokenizers | `tiktoken_wrap`, `hf_tokenizer`, `byte_tokenizer` |
| Scalers | `amp_grad_scaler`, `keller_scaler` |

---

## Training Pipeline

The training pipeline (`nanugpt/train.py`) follows a structured flow:

### Initialization Phase

```
1. Load and process config
2. Setup device (CPU/CUDA) and distributed training
3. Initialize logger (console + W&B)
4. Load data (train/val/test loaders)
5. Create tokenizer
6. Instantiate model with loss wrapper
7. Apply DDP wrapper (if distributed)
8. Apply torch.compile (if enabled)
9. Create optimizer and scheduler
10. Setup gradient scaler
11. Save initial artifacts (code, config)
```

### Training Loop

```python
for step in range(max_steps):
    # Gradient accumulation loop
    for micro_step in range(grad_acc_steps):
        x, y = batches.next()           # Get batch (infinite iterator)

        with amp_ctx:                    # Mixed precision context
            _, loss, correct = model(x, labels=y, return_logits=False)

        loss = loss / grad_acc_steps    # Scale for accumulation
        scaler.backward(loss)            # Accumulate gradients

        # Sync gradients only on last micro-step (DDP optimization)
        model.require_backward_grad_sync = (micro_step == grad_acc_steps - 1)

    # Gradient clipping
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    # Optimizer step
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    scheduler.step()

    # Evaluation (periodic)
    if step % eval_every == 0:
        val_loss, val_acc = estimate_loss(model, val_loader)

    # Checkpointing (time-based or final)
    if should_checkpoint:
        save_checkpoint(model, optimizer, scheduler, step)

    # Logging
    logger.info(metrics)
```

### The Batches Class

Provides infinite iteration over a finite dataloader:

```python
class Batches:
    def __init__(self, loader):
        self.loader = loader
        self.iter = iter(loader)

    def next(self):
        try:
            return next(self.iter)
        except StopIteration:
            self.iter = iter(self.loader)  # Auto-restart
            return next(self.iter)
```

This eliminates epoch-based logic in favor of step-based training.

### ModelWithLoss Wrapper

Combines model and loss computation for torch.compile optimization:

```python
class ModelWithLoss(nn.Module):
    def __init__(self, model: nn.Module, get_loss: Callable):
        self.model = model
        self.get_loss = get_loss

    def forward(self, input_ids, labels=None, return_logits=True):
        logits = self.model(input_ids)

        if labels is not None:
            loss, correct = self.get_loss(logits, labels)
            return (logits if return_logits else None), loss, correct

        return logits, None, None
```

Benefits:
- **torch.compile fusion**: Loss computation fused with backward pass (~30% speedup)
- **Memory optimization**: `return_logits=False` frees memory during training
- **Clean interface**: Single forward returns all needed values

---

## Data Pipeline

### Memory-Mapped Data Loading

For large datasets, nanuGPT uses memory-mapped files:

```
Raw Text → Tokenization → .bin file (memory-mapped) → MemmapDataset → MemmapDataloader
```

#### Tokenization (`tokenize_dataset.py`)

```bash
python tokenize_dataset.py configs/tokenize/tinyshakespeare_byte.yaml
```

Produces `.bin` files with tokenized sequences stored as numpy arrays.

#### MemmapDataset

```python
class MemmapDataset(Dataset):
    def __init__(self, data: np.ndarray, context_length: int):
        self.data = data  # Memory-mapped numpy array
        self.context_length = context_length
        self.seq_count = len(data) - context_length + 1

    def __getitem__(self, idx):
        return self.data[idx:idx + self.context_length]
```

- Random access into tokenized data
- No loading entire dataset into memory
- Efficient for datasets larger than RAM

#### MemmapDataloader

Custom dataloader with:
- Configurable shuffling with seed
- Batch size handling
- Distributed training support (offset-based data splitting)

### HuggingFace Dataset Support

For on-the-fly tokenization:

```yaml
data:
  module: 'nanugpt.data.hf_dataset.get_data'
  module_kwargs:
    hf_name_path: 'wikitext'
    hf_name: 'wikitext-103-raw-v1'
```

---

## Distributed Training

nanuGPT supports multi-GPU training via PyTorch's DistributedDataParallel (DDP).

### Launch Methods

#### Local Multi-GPU

```bash
torchrun --nproc_per_node=8 --standalone train.py config.yaml
```

#### Slurm Cluster

```bash
NODES=2 bash ./scripts/slurm/sbatch_ex.sh train.py config.yaml
```

### Environment Variables

Set automatically by `torchrun`:

| Variable | Description |
|----------|-------------|
| `RANK` | Global rank across all nodes/GPUs (0 = master) |
| `LOCAL_RANK` | Rank within single node |
| `WORLD_SIZE` | Total number of processes |
| `LOCAL_WORLD_SIZE` | Processes per node |

### TorchInfo Dataclass

Tracks distributed state:

```python
@dataclass
class TorchInfo:
    is_accelerator: bool      # Has GPU
    is_distributed: bool      # Multi-GPU training
    device_type: str          # 'cuda' or 'cpu'
    dtype: str                # 'bfloat16', 'float16', 'float32'
    device_id: int            # GPU index for this process
    device_name: str          # 'cuda:0', 'cuda:1', etc.
    global_rank: int          # Global rank
    local_rank: int           # Local rank
    world_size: int           # Total processes
    is_master: bool           # global_rank == 0
```

### DDP Wrapping

```python
if torch_info.is_distributed:
    model = DistributedDataParallel(
        model,
        device_ids=[torch_info.device_id],
        gradient_as_bucket_view=True  # Memory optimization
    )
```

### Data Distribution

Each rank processes different data via offset-based splitting:

```python
# Non-overlapping contiguous chunks
train_offset = int((len(dataset)-1) * float(global_rank) / world_size)
```

### Gradient Synchronization Optimization

Gradients synchronized only on final micro-step of gradient accumulation:

```python
model.require_backward_grad_sync = (micro_step == grad_acc_steps - 1)
```

This reduces communication overhead by (grad_acc_steps - 1) all-reduce operations per step.

### Metric Reduction

All metrics reduced to rank 0 for logging:

```python
if torch_info.is_distributed:
    dist.reduce(metrics_tensor, dst=0, op=dist.ReduceOp.SUM)
    dist.barrier()
```

---

## Mixed Precision & Gradient Accumulation

### Mixed Precision (AMP)

Automatic Mixed Precision reduces memory and increases speed:

```python
# Setup
amp_ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)

# Usage
with amp_ctx:
    _, loss, correct = model(x, labels=y)
```

#### Gradient Scaling (float16 only)

For float16 (not bfloat16), gradient scaling prevents underflow:

```python
scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))

# Training step
scaler.scale(loss).backward()      # Scale loss before backward
scaler.unscale_(optimizer)         # Unscale before clipping
clip_grad_norm_(model, grad_clip)  # Clip at correct magnitude
scaler.step(optimizer)             # Step with scaled gradients
scaler.update()                    # Adjust scale for next iteration
```

### Gradient Accumulation

Simulates larger batch sizes without memory overhead:

```python
# Effective batch size
global_batch_size = grad_acc_steps × device_batch_size × world_size

# Computation
grad_acc_steps = round((global_batch_size / device_batch_size) / world_size)
```

Each micro-step:
1. Forward pass with micro-batch
2. Scale loss by `1/grad_acc_steps`
3. Backward pass (accumulates gradients)

After all micro-steps:
1. Unscale gradients
2. Clip gradients
3. Optimizer step
4. Zero gradients

---

## Checkpoint System

### Checkpoint Contents

```python
checkpoint = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'scheduler': scheduler.state_dict(),
    'train/step': step,
    'val/best_loss': best_val_loss
}
```

### Saving Strategy

- **Time-based**: Save every N hours (configurable)
- **Best model**: Save when validation loss improves
- **Final**: Save at end of training

Only rank 0 saves checkpoints in distributed training.

### Checkpoint Log

`checkpoint_log.yaml` tracks all saved checkpoints:

```yaml
- step: 1000
  path: /out/checkpoint_1000.pt
  val_loss: 4.23
- step: 2000
  path: /out/checkpoint_2000.pt
  val_loss: 3.87
```

### Loading Checkpoints

```python
checkpoint = torch.load(path, weights_only=True)
state_dict = checkpoint['model']

# Handle torch.compile prefix
state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

model.load_state_dict(state_dict)
```

---

## Logging System

The logging system (`nanugpt/glogging.py`) provides dual-track logging.

### Logger Initialization

```python
logger = Logger(
    project_name='my_project',
    run_name='experiment_1',
    enable_wandb=True,
    log_dir='./logs',
    is_master=torch_info.is_master
)
```

### Console Logging

- Color-coded output via Rich
- Metrics formatted as `key=value` pairs
- Non-master ranks log at WARNING level only

### File Logging

- All logs written to `{out_dir}/log.txt`
- Includes timestamps and log levels

### Weights & Biases Integration

```python
# Summary metrics (shown in W&B overview)
logger.summary({
    "run/global_batch_size": 512,
    "model/params": 124_000_000,
})

# Step metrics (plotted over time)
logger.info({
    "train/loss": 2.45,
    "train/acc": 0.42,
    "run/lr": 3e-4,
})
```

### Artifact Logging

```python
logger.log_artifact(
    name='config',
    type='yaml',
    file_or_dir=config_path,
    desc_markdown="Training configuration"
)
```

### Key Metrics

| Metric | Description |
|--------|-------------|
| `train/loss` | Training loss per step |
| `train/acc` | Training accuracy |
| `train/ppl` | Perplexity (exp(loss)) |
| `train/tokens_per_sec` | Training throughput |
| `train/pre_clip_norm` | Gradient norm before clipping |
| `val/loss` | Validation loss |
| `val/generalization_gap` | val_loss - train_loss |
| `run/lr` | Current learning rate |
| `run/eta_hr` | Estimated time to completion |

---

## Generation & Inference

The generation system (`nanugpt/generate.py`) provides text generation from trained models.

### Generator Class

```python
generator = Generator(
    config=config,
    checkpoint_dir='/path/to/checkpoints'
)

results = generator.generate(
    prompts=["Once upon a time"],
    max_length=100,
    temperature=0.8,
    top_k=50
)
```

### Checkpoint Discovery

1. Look for `checkpoint_log.yaml` in specified directory
2. Find latest checkpoint by modification time
3. Load model state dict

### Autoregressive Decoding

```python
for _ in range(max_length):
    # Crop to context length
    idx_cond = idx[:, -context_length:]

    # Forward pass
    logits = model(idx_cond)
    logits = logits[:, -1, :]  # Last position only

    # Temperature scaling
    logits = logits / temperature

    # Optional top-k filtering
    if top_k:
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[:, [-1]]] = float('-inf')

    # Sample
    probs = F.softmax(logits, dim=-1)
    idx_next = torch.multinomial(probs, num_samples=1)

    # Append
    idx = torch.cat([idx, idx_next], dim=1)
```

### Usage

```bash
python generate.py configs/train_gpt2/tinyshakespeare.yaml
```

---

## Key Files Reference

### Entry Points

| File | Purpose |
|------|---------|
| `train.py` (root) | Training entry point, loads config and calls `nanugpt.train.train()` |
| `generate.py` (root) | Generation entry point |
| `tokenize_dataset.py` | Tokenization entry point |

### Core Library

| File | Purpose |
|------|---------|
| `nanugpt/train.py` | Main training loop, `train()` function |
| `nanugpt/generate.py` | Generator class for inference |
| `nanugpt/config.py` | YAML configuration with inheritance |
| `nanugpt/common.py` | Setup utilities, device/data/model creation |
| `nanugpt/utils.py` | General utilities, distributed helpers |
| `nanugpt/glogging.py` | Logging system with W&B integration |

### Plugin Directories

| Directory | Contents |
|-----------|----------|
| `nanugpt/models/` | Model implementations (GPT, LLaMA, etc.) |
| `nanugpt/data/` | Data loading implementations |
| `nanugpt/optimizers/` | Optimizer implementations |
| `nanugpt/schedulers/` | LR scheduler implementations |
| `nanugpt/losses/` | Loss function implementations |
| `nanugpt/tokenizers/` | Tokenizer implementations |
| `nanugpt/scalers/` | Gradient scaler implementations |

### Configuration

| Directory | Contents |
|-----------|----------|
| `configs/base_config.yaml` | Global defaults |
| `configs/train_gpt2/` | GPT-2 training configs |
| `configs/train_llama2/` | LLaMA training configs |
| `configs/grokking/` | Grokking experiment configs |
| `configs/tokenize/` | Tokenization configs |

### Scripts

| Directory | Contents |
|-----------|----------|
| `scripts/slurm/` | Slurm cluster scripts |
| `scripts/volcano/` | Kubernetes/Volcano scripts |

---

## Quick Reference

### Running Training

```bash
# Single GPU
python train.py configs/train_gpt2/tinyshakespeare.yaml

# Multi-GPU (local)
torchrun --nproc_per_node=8 --standalone train.py config.yaml

# With overrides
python train.py config.yaml --training.max_steps 10000 --optimizer.module_kwargs.learning_rate 1e-4
```

### Adding a New Plugin

1. Create file in appropriate directory (e.g., `nanugpt/models/my_model.py`)
2. Implement factory function following interface contract
3. Reference in config: `module: 'nanugpt.models.my_model.get_model'`

### Debugging

- Use VSCode debug configurations (dropdown next to play button)
- Set breakpoints in training loop
- Check `log.txt` in output directory for full logs

### Common Issues

| Issue | Solution |
|-------|----------|
| CUDA OOM | Reduce `device_batch_size`, increase `grad_acc_steps` |
| Slow training | Enable `torch_compile: true` in config |
| NaN loss | Check learning rate, enable gradient clipping |
| Distributed hang | Check NCCL environment, try `NCCL_DEBUG=INFO` |
