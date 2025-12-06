# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

nanuGPT is a modular, hackable GPT training framework derived from Karpathy's nanoGPT. It prioritizes modularity and scalability for experiments while maintaining simplicity.

## Common Commands

### Installation
```bash
pip install -e .
```

### Training
```bash
# Single GPU
python train.py configs/train_gpt2/tinyshakespeare.yaml

# Multi-GPU (local)
torchrun --nproc_per_node=8 --standalone train.py configs/train_gpt2/tinyshakespeare.yaml

# Slurm cluster
NODES=1 DATA_ROOT=<dir> OUT_DIR=<dir> JOB_NAME=<name> bash ./scripts/slurm/sbatch_ex.sh train.py <config.yaml>
```

### Tokenization
```bash
python tokenize_dataset.py configs/tokenize/tinyshakespeare_byte.yaml
```

### Generation
```bash
python generate.py configs/train_gpt2/tinyshakespeare.yaml
```

### Testing
```bash
pytest tests/
pytest tests/test_memmap_dataloader.py  # single test file
```

## Environment Variables

Required:
- `DATA_ROOT`: Directory for datasets
- `OUT_DIR`: Directory for output/checkpoints

Optional:
- `WANDB_API_KEY`: For Weights & Biases logging
- `WANDB_HOST`: Custom W&B host

## Architecture

### Configuration System (`nanugpt/config.py`)
- YAML-based with inheritance via `__include__`
- Supports value copying with `_copy: /path/to/value`
- Environment variable expansion with `$VAR` syntax
- Command-line overrides: `--section.key value`
- Time stamps: `_time:` directive

### Plugin Architecture
Components are specified in config as `module` + `module_kwargs`:
- **Models**: `nanugpt/models/` - nanogpt, hf_gpt2, hf_llama, tinyllama, te_llama3
- **Data**: `nanugpt/data/` - tokenized_data, hf_dataset, grokking_data, arithmatic_data
- **Optimizers**: `nanugpt/optimizers/` - adamw, adamw_nanogpt, muon
- **Schedulers**: `nanugpt/schedulers/` - cosine, linear, constant, lr_range_test
- **Losses**: `nanugpt/losses/` - autoregressive_loss, grokking_loss, arithmatic_loss
- **Tokenizers**: `nanugpt/tokenizers/` - tiktoken_wrap, hf_tokenizer, byte_tokenizer
- **Scalers**: `nanugpt/scalers/` - amp_grad_scaler, keller_scaler

### Training Loop (`nanugpt/train.py`)
- Gradient accumulation for large effective batch sizes
- Mixed precision (bfloat16/float16) with AMP
- Distributed training via DDP with NCCL backend
- Checkpoint saving with configurable intervals
- Integrated W&B logging

### Key Config Sections
```yaml
general:     # project_name, out_dir, device_type, torch_compile, seed, dtype
model:       # module path + n_layer, n_embd, n_head, context_length
data:        # module path + tokenized paths, batch sizes
training:    # device_batch_size, global_batch_size, max_steps, grad_clip
optimizer:   # module path + learning_rate, weight_decay, betas
scheduler:   # module path + warmup_iters, max_iters
eval:        # eval_every, eval_iters, checkpoint settings
```

### Config Directory Structure
- `configs/train_gpt2/` - GPT-2 training configs
- `configs/train_llama2/` - LLaMA training configs
- `configs/grokking/` - Grokking experiment configs
- `configs/tokenize/` - Tokenization configs

## Data Pipeline

1. Raw data → Tokenization (`tokenize_dataset.py`) → `.bin` files (memory-mapped)
2. `MemmapDataset` provides random access to tokenized sequences
3. `MemmapDataloader` handles batching with configurable shuffling
