import os
from datasets import load_dataset
from nanugpt import utils

ds = load_dataset("iohadrubin/wikitext-103-raw-v1")
output_dir = utils.full_path("$DATA_ROOT/datasets/wikitext-103-raw-v1")
os.makedirs(output_dir, exist_ok=True)
print(f"Saving dataset to {output_dir}")
ds.save_to_disk(output_dir) # type: ignore