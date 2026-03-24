# python scripts/datasets/climbmix/download.py
#
# Downloads OptimalScale/ClimbMix dataset and saves as arrow files.
# If the dataset is already in the HF cache, it loads from there
# without re-downloading, then saves to disk in arrow format.

from datasets import load_dataset

from nanugpt import utils

# Specify the folder where the dataset will be saved
save_directory = "$DATA_ROOT/datasets/climbmix"

# resolve env vars
save_directory = utils.full_path(save_directory, create=True)

# Load dataset from HF (uses existing HF cache if available, avoids re-download)
dataset = load_dataset("OptimalScale/ClimbMix")

# Print the dataset structure
print(dataset)

# Save as arrow files so load_from_disk() can be used downstream
dataset.save_to_disk(save_directory)
print(f"Saved to {save_directory}")

