# Training on Tinyshakespeare

You can train on Tinystories V2 dataset like this:

```bash
# download raw dataset
mkdir -p $DATA_ROOT/datasets/tinyshakespeare
wget -P $DATA_ROOT/datasets/tinyshakespeare https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# tokenize input file using byte tokenizer
python tokenize_dataset.py configs/tokenize/tinyshakespeare_byte.yaml

# run training using GPT2 124M model
python train.py configs/train_gpt2/tinyshakespeare.yaml

# generate completitions for the prompt
python generate.py configs/train_gpt2/tinyshakespeare.yaml
```

![Training and Validation Loss](results/grokking/prime223/prime223_baseline.png)
Stats:

```text
Model params (all): 10,819,968
Model params (non emb): 10,621,824
Train tokens: 40,960,000
Train samples: 160,000
Global batch size: 512
Train steps: 2,499
Context length: 256
Train loss: 1.0899850130081177
Val loss: 1.2700303447246553
Run time: 0.02099152487363123 hr (1x NVIDIA A100 80GB PCIe)
```

[Detailed Logs](results/grokking/prime223/log.txt)