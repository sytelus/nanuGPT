# Welcome to nanuGPT

This repository contains mostly just Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) but has more organization to support plug-n-play datasets, optimizers, schedulers and models, especially from HuggingFace.

The code is kept in line with Karpathy's original philosophy of keeping things simple and hackable so anyone can do quick experiments with transformers.

This playground is for GPU poor. Using tiny datasets, I often do experimental training runs in just 10 minutes on single RTX 3080. There is no support of doing large training runs or models beyond a single node.

The name `nanu` in Gujarati language means small but not super small :stuck_out_tongue_winking_eye:.

## How to Install

```python
git clone https://github.com/sytelus/nanuGPT
cd nanuGPT
pip install -e .
```

## Setting Up Data and Output Directories

To use many of the config files as-is, you should create environment variables for your data and output directories:

```bash
export DATA_ROOT=<my data dir>
export OUT_DIR=<my output dir>
```

Tiny shakespear is good quick test dataset. You can download it as follows:

```bash
mkdir -p $DATA_ROOT/datasets/tinyshakespeare
wget -P $DATA_ROOT/datasets/tinyshakespeare https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

## How to Use

### Synthetic Arithmatic Dataset

You can reproduce [Grokking](https://arxiv.org/abs/2201.02177) phenomenon in just 10 minutes of training on a single RTX 3080. Here, we train a tiny transformer that can learn to compute a simple math expression reaching train and eventually val loss of ~0. Synthetic dataset is generated and tokenized on the fly. To try this, run:

```python
python train.py configs/grokking/prime223.yaml
```

### TinyShakespear Dataset

You can also train a language model using all the works of Shakespear as data in 5 minutes on single RTX 3080 just like in original NanoGPT using this:

```python
# tokenize input file using byte tokenizer
python tokenize_dataset.py configs/tokenize/tinyshakespeare.yaml

# run training using GPT2 124M model
python train.py configs/train_gpt2/tinyshakespeare.yaml

# generate completitions for the prompt
python generate.py cconfigs/train_gpt2/tinyshakespeare.yaml
```

### Other Datasets

For lucky people with more compute, there are configs available to replicate benchmarks on WikiText103, TinyStories and OpenWebText.

### Config

All config files can be found in `configs/` and things are self-explanatory. Sometime you might notice the top line with `__include__:` to specify the base config that you can inherit from. This is a custom feature implemented in this repo so we can share the base config across many experiments. The `base_config.yaml` serves as defaults that can be overriden in your yaml.

### Tokenization

The code uses all cores and achieves 157k tokens/sec/core end-to-end. This corresponds to tokenizing OpenWebText in roughly 0.5hr on 128 core machine.

### Logging

NanuGPT vastly improves on logging, i.e., we log everything we can! The idea is that once run is done, you can examine the log to do lot of post-hoc debugging. You can also enable Weights and Biases (wandb) by enabling in config and creating environment variable for `WANDB_API_KEY`.

The consol logs are colored for quick glances.

### Debugging

If you are using VSCode, please do take advantage of dropdown next to play button run any config in debug mode, set breakpoints, look at variables and enjoy!


## Credits

This repository uses code from [nanoGPT](https://github.com/karpathy/nanoGPT) and [grokking](https://github.com/danielmamay/grokking) as foundation. The code here is inspired from the philosophy and style of these authors. Beyond that, you might find very little novelty here.
