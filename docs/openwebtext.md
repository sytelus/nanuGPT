# Training on OpenWebText

You can train on OpenWebText dataset like this:

```bash
# download dataset
python scripts/datasets/openwebtext/download.py

# tokenize input file using byte tokenizer
python tokenize_dataset.py configs/tokenize/openwebtext_tiktoken_gpt2.yaml

# training using GPT2 124M model
python train.py configs/train_gpt2/openwebtext.yaml

# generate completitions for the prompt
python generate.py configs/train_gpt2/openwebtext.yaml
```

![Training and Validation Loss](results/openwebtext/openwebtext_baseline.png)
Stats:

```text
Model params (all): 124,337,664
Model params (non emb): 84,953,856
Dataset train tokens: 998,021
Dataset val tokens: 110,153
Dataset train samples: 6499
Dataset val samples: 723
Vocab Size: 260
Trained on total tokens: 36,550,717,440
Global batch size: 64
Train steps: 2500
Context length: 256
Train loss: 1.0831478834152222
Val loss: 1.2721147787570954
Run time: 0.016054142607241 hr (1x NVIDIA A100 80GB PCIe)
```

# Reproducibility

Karpathy's original training runs for 600,000 iterations (294B tokens, ~32 epochs) consuming 43.2hr on 8xA100/40GB without `torch.compile` and 22.5hr with `torch.compile`, reaching val loss of 2.85.