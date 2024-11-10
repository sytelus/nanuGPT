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
python tokenize_dataset.py configs/tokenize/tinyshakespeare.yaml

# run training using GPT2 124M model
python train.py configs/train_gpt2/tinyshakespeare.yaml

# generate completitions for the prompt
python generate.py cconfigs/train_gpt2/tinyshakespeare.yaml
```

Above will train model using all the works of Shakespear as data in 5 minutes on single RTX 3080 just like in original NanoGPT.

## Training on Tinystories

### Synthetic Arithmatic Dataset

You can reproduce [Grokking](https://arxiv.org/abs/2201.02177) phenomenon in just 10 minutes of training on a single RTX 3080. Here, we train a tiny transformer that can learn to compute a simple math expression reaching train and eventually val loss of ~0. Synthetic dataset is generated and tokenized on the fly. To try this, run:

```python
python train.py configs/grokking/prime223.yaml
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
