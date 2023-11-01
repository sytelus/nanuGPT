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

## How to Use

As an example, you can reproduce [Grokking](https://arxiv.org/abs/2201.02177) phenomenon in just 10 minutes of training on a single RTX 3080. Here, we train a tiny transformer that can learn to compute a simple math expression reaching train and eventually val loss of ~0. Synthetic dataset is generated and tokenized on the fly. To try this, run:

```python
python nanugpt/train.py configs/grokking/prime223.yaml
```

You can also train a language model using all the works of Shakespear as data in 5 minutes on single RTX 3080 just like in original NanoGPT using this:

```python
# tokenize input file using byte tokenizer
python nanugpt/tokenize_dataset.py configs/tokenize/tinyshakespeare.yaml

# run training using GPT2 124M model
python nanugpt/train.py configs/train_llm/tinyshakespeare.yaml

# generate completitions for the prompt
python nanugpt/generate.py cconfigs/train_llm/tinyshakespeare.yaml
```

For lucky people with more compute, there are configs available to replicate benchmarks on WikiText103, TinyStories and OpenWebText.

If you are using VSCode, please do take advantage of dropdown next to play button run any config in debug mode, set breakpoints, look at variable and enjoy!

## Credits

This repository uses code from [nanoGPT](https://github.com/karpathy/nanoGPT) and [grokking](https://github.com/danielmamay/grokking) as foundation. The code here is inspired from the philosophy and style of these authors. Beyond that, you might find very little novelty here.
