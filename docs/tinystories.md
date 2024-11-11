# Training on Tinystories

You can train on Tinystories V2 dataset like this:

```bash
# download raw datasets and convert it to Arrow format
bash scripts/datasets/tinystories/download.sh
python scripts/datasets/tinystories/tinystories2arrow.py

# tokenize using GPT2 tokenizer
python tokenize_dataset.py configs/tokenize/tinystories_tiktoken_gpt2.yaml

# run training using GPT2 124M model
python train.py configs/train_gpt2/tinystories.yaml

# generate completitions for the prompt
python generate.py configs/train_gpt2/tinystories.yaml
```

![Training and Validation Loss](results/grokking/prime223/prime223_baseline.png)
Stats:

```text
Model params (all): 457,578
Model params (non emb): 426,986
Train tokens: 30,309,625
Train samples: 24,753
Global batch size: 512
Train steps: 12,000
Context length: 5
Train loss: 0.00000010128132288401
Val loss: 0.00000022627273210674
Run time: 0.14156427994833293 hr (1x NVIDIA RTX 4500 Ada Generation)
```

[Detailed Logs](results/grokking/prime223/log.txt)