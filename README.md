# Welcome to nanuGPT

This repository contains heavily modified version of Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT). The code is kept in line with Karpathy's original philosophy of keeping things simple and hackable so anyone can do quick and reproducible experiments however we break away from the philosophy of reducing LoC and instead adopt obsession towards modularity and scaling ability to do experiments while keeping sanity.

Name of this repo derives from `nanu` which is a word in Gujarati language meaning "small but not super small" :stuck_out_tongue_winking_eye:.

Some of the features this code base adds:

* Moduler approach
* Plug-n-Play datasets
  * Plug-n-Play optimizers
  * Plug-n-Play schedulers
  * Plug-n-Play models
* Full yaml based config
  * Inherit from base config for experiments
  * Override any settings from command line
  * Share common config between modules
* Play nicely with other stacks
  * Use PyTorch style Dataset, DataLoader, schedulers
  * Support HuggingFace datasets
  * Full WandB support
* Support Grokking experiments
  * Fully reproducible
  * So fast, no GPUs needed!
  * Uses exact same code as big models
* New datasets
  * tinystories
  * Grokking
* New models
  * llama
* Other featues
  * Detailed local and WandB logging
  * On-the-fly tokenization for HF datasets
  * Byte tokenizer
  * Everything obsessively tested
  * Confirmed to work with A6000, RTX 3090/4090, A100, H100, GH200

This code base is designed for GPU poor. Using tiny datasets, I often do experimental training runs in just 10 minutes on single RTX 3080. For Grokking experiments, you can even do CPU-only runs!!

## Documentation

| Document | Description |
|:---------|:------------|
| [OVERVIEW.md](OVERVIEW.md) | Technical overview of the codebase architecture, training pipeline, distributed training, and all major components |
| [STYLE.md](STYLE.md) | Coding style guide and conventions for contributors |
| [CLAUDE.md](CLAUDE.md) | Quick reference for AI assistants and common commands |

## How to Install

```python
git clone https://github.com/sytelus/nanuGPT
cd nanuGPT
pip install -e .
```

## How To Use

Create environment variables:

```bash
export DATA_ROOT=<my data dir>
export OUT_DIR=<my output dir>

# OPTIONAL
export WANDB_API_KEY=<YOUR_KEY>
export WANDB_HOST=<YOUR_HOST>
```

Karpathy's Tiny shakespear is good quick dataset for testing. You can download it as follows:

```bash
mkdir -p $DATA_ROOT/datasets/tinyshakespeare
wget -P $DATA_ROOT/datasets/tinyshakespeare https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

You can then tokenize, train and generate from the model like this:

```python
# tokenize input file using byte tokenizer
python tokenize_dataset.py configs/tokenize/tinyshakespeare_byte.yaml

# run training using GPT2 124M model
python train.py configs/train_gpt2/tinyshakespeare.yaml

# generate completitions for the prompt
python generate.py configs/train_gpt2/tinyshakespeare.yaml
```

Above will train model using all the works of Shakespear as data in 5 minutes on single RTX 3080 just like in original NanoGPT. See [Tinyshakespeare baseline](docs/tinyshakespeare.md).

## Using Multi GPUs

### Local Node

To run on multiple GPUs on local node, instead of `python` you can use `torchrun` like this:

```bash
torchrun --nproc_per_node=8 --standalone train.py configs/train_gpt2/tinyshakespeare.yaml
```

### Slurm Cluster

If you are working in slurm environment, you can also run multinode job like this:

```bash
NODES=1 DATA_ROOT=<my_data_dir> OUT_DIR=<my_output_dir> JOB_NAME=<my_job> \
bash ./scripts/slurm/sbatch_ex.sh train.py configs/train_gpt2/openwebtext_classic.yaml

# also see script slurm_owt10b_baseline.sh for more robustness and flexibility
```

Above command uses Nvidia NGC image by default, mounts directory specified in `DATA_ROOT` and `OUT_DIR` on the container, creates job sub directory with datatime and then launches specified command line using python for each worker process (default 8 processes for 8 GPUs per node) in each node.

Please see `sbatch_ex.sh` for various options offered by this script.

### Volcano Cluster

For Volcano/k8s infrastructure, you can launch multigpu job like this:

```bash
bash volcano_karpathy_owt10b.sh
```

## Using Other Datasets

[How to train on Tinystories](docs/tinystories.md)

[How to train on WikiText-103](docs/wt103.md)

[How to train on WikiText-103](docs/wt103.md)

[How to train on OpenWebText](docs/openwebtext.md)

[How to train on Grokking dataset](docs/grokking.md)

### Config

All config files can be found in `configs/` and things are self-explanatory. Sometime you might notice the top line with `__include__:` to specify the base config that you can inherit from. This is a custom feature implemented in this repo so we can share the base config across many experiments. The `base_config.yaml` serves as defaults that can be overriden in your yaml.

#### Various Available Configs

| Config                   | Description                                 |
|:-------------------------|:--------------------------------------------|
| configs/train_gpt2/openwebtext_classic.yaml | NanoGPT 124M params 295B tokens run using original hyper params by Karpathy |
| configs/train_gpt2/openwebtext_tokens10b_classic.yaml | 124M params 10B tokens run using original NanoGPT hyper params by Karpathy |
| configs/train_gpt2/openwebtext_tokens10b_karpathy_llmc.yaml | 124M params 10B tokens run using llm.c hyper params by Karpathy |
| configs/train_gpt2/openwebtext_tokens10b_keller_adamw.yaml | 124M params 10B tokens run using AdamW+WSD hyper params by Keller Jordan |
| configs/train_gpt2/tinyshakespeare.yaml | Training using TinyShakespeare dataset |
| configs/train_gpt2/tinystories.yaml | Training using Tinystories dataset |
| configs/train_gpt2/wikitext103.yaml | Training using WikiText-103 dataset |
| configs/grokking/baseline.yaml | Reproduces grokking paper results  |
| configs/grokking/prime50k.yaml | Grokking experiment with larger prime for more data  |
| configs/grokking/prime223.yaml | Grokking experiment that runs much faster than original paper |

### Tokenization

The code uses all cores and achieves 65k tokens/sec/core end-to-end on Intel Xenon w5-3425 and 223k tokens/sec/core on Neoverse-v2. Typically you can tokenize OpenWebText in roughly 0.5hr on 128 core server-grade machine.

### Logging

NanuGPT has huge focus on detailed logging, i.e., log everything we can! The idea is that once run is done, you can examine the log to do lot of post-hoc debugging. You can also enable Weights and Biases (wandb) by enabling in config and creating environment variable for `WANDB_API_KEY` and optionally `WANDB_HOST`.

The consol logs are colored for quick glances.

### Debugging

If you are using VSCode, please do take advantage of dropdown next to play button run any config in debug mode, set breakpoints, look at variables and enjoy!

## Credits

This repository uses code from [nanoGPT](https://github.com/karpathy/nanoGPT) and [grokking](https://github.com/danielmamay/grokking) as starting point. The code here is inspired from the philosophy and style of these authors.
