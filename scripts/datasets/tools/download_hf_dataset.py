# reference:
# - https://huggingface.co/docs/datasets/v3.2.0/en/package_reference/loading_methods#datasets.load_dataset
# - https://huggingface.co/docs/hub/datasets-usage

from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("bespokelabs/Bespoke-Stratos-17k")