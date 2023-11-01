# Welcome to nanuGPT

This repository contains mostly just Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT), slighly more organized and extended. The code is kept in line with Karpathy's original philosophy of keeping things simple, readable, reliabe, and, *most importantly*, hackable so anyone can do quick experiments with transformers.

Many of my training runs lasts about a minute on single RTX 3080, so this is ideal playground for GPU poor. There is no support of doing large training runs or models beyond a single node.

There is also some divergence from nanoGPT because more structure is added at the risk of not achieving minimum lines of code. This is the reason we adopt the name `nanu` which in Gujarati language means small but not really that small :stuck_out_tongue_winking_eye:.

# How to Install

We don't discriminate based on your operating system. NanuGPT works on Linux, Windows and OSX. Just make sure you have Python and PyTorch installed.

```
git clone https://github.com/sytelus/nanuGPT
cd nanuGPT
pip install -e .
```

# How to Use

NanuGPT is for GPU poor. To ensure it stays that way, I do most of the testing on single RTX 3080. Much of the time I expect training to take less than a minute. Sometimes it takes 5 minutes which, of course, is bit too much.

You might ask what kind of training is that? Well, it's bassed on [Grokking paper](https://arxiv.org/abs/2201.02177). In that paper authors train a tiny transformer that can learn to compute a simple math expression. So, the entire dataset is synthetic, generated on the fly. To try this, run:

```
python nanugpt\train.py configs/grokking/prime223.yaml
```

If you got more time, like 5 minutes of single RTX 3080, try training a language model using all the works of Shakespear like this:

```
# tokenize input file using byte tokenizer
python nanugpt\tokenize_dataset.py configs/tokenize/tinyshakespeare.yaml

# run training using GPT2 124M model
python nanugpt\train.py configs/train_llm/tinyshakespeare.yaml

# generate completitions for the prompt
python nanugpt\generate.py cconfigs/train_llm/tinyshakespeare.yaml
```

For lucky people with more compute, there are configs available to replicate benchmarks on WikiText103, TinyStories and OpenWebText.

If you are using VSCode, please do take advantage of dropdown next to play button run any config in debug mode, set breakpoints, look at variable and enjoy!

# Credits

This repository uses code from [nanoGPT](https://github.com/karpathy/nanoGPT) and [grokking](https://github.com/danielmamay/grokking) as foundation. The code here is heavily inspired from the philosophy and style set by these authors. Beyond that, you might find only a little novelty here.
