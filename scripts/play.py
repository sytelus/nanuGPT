import os

from transformers import AutoTokenizer
import transformers
import torch
import numpy as np
from gptplay.tokenizers.byte_tokenizer import ByteTokenizer

import os
path=os.path.dirname("C:/folder1/folder2/filename.xml")
print(path)
print(os.path.basename(path))

bt = ByteTokenizer()
en = bt.batch_encode(["ab<EOS>c", "<UNK>def"])
print(en)
de = bt.batch_decode(en["input_ids"])
print(de)

n = np.array([1, 5, 8, 2, 5, 7, 2, 3])
ind = np.where(n == 5)

s = bytes([]).decode(encoding="utf-8")

d = {"a": 1, "b": 2}

b = "abc".encode("utf-8")
i = list(b)
print(i)
