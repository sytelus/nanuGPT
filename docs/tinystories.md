## Training on Tinystories

First download row datasets, convert it to Arrow format:

```bash
bash scripts/datasets/tinystories/download.sh
python scripts/datasets/tinystories/tinystories2arrow.py
```

You can then tokenize, train and generate from the model like this:

```python
# tokenize using GPT2 tokenizer
python tokenize_dataset.py configs/tokenize/tinystories_tiktoken_gpt2.yaml

# run training using GPT2 124M model
python train.py configs/train_gpt2/tinystories.yaml

# generate completitions for the prompt
python generate.py configs/train_gpt2/tinystories.yaml
```
