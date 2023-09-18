import os

from transformers import AutoTokenizer
import transformers
import torch

model = "Salesforce/codegen-350M-nl"

tokenizer = AutoTokenizer.from_pretrained(model, use_auth_token=os.environ.get('HF_AUTH_TOKEN', None))

print(tokenizer.pad_token)
