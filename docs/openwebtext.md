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

![Training and Validation Loss](results/openwebtext/loss-log_step.png)

Stats:

```text
Model params (all): 124,337,664
Model params (non emb): 84,953,856
Dataset train tokens: 9,035,525,879
Dataset val tokens: 4,489,170
Dataset train samples: 752,960,490
Dataset val samples: 140,287
Vocab Size: 50,257
Trained on total tokens: 294,912,000,000
Global batch size: 480
Train steps: 600,000
Context length: 1024
Train loss: 2.7935577392578126
Val loss: 2.83925675034523
Run time: 61.71 hr (8x NVIDIA H100 80GB HBM3)
```

## Reproducibility

Karpathy's original GPT2-124M run does 600,000 steps on a global batch of 480 with context length of 1024. So, that's 294B tokens in training, about 32.8 epochs. This would should take 1.24 days per training calculator (237.68 hours) on 8xH100/BF16 (~4 days on A100). Run uses default settings by Karpathy (LR of 6E-4 and warmup of 2000 etc). Val loss is 2.85 (OpenAI GPT2 val loss is 3.11 due to distribution differences, train loss 3.12).
