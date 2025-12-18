from datasets import load_dataset

from nanugpt import utils

# Specify the folder where the dataset will be saved
save_directory = "$DATA_ROOT/datasets/openwebtext"

# resolve env vars
save_directory = utils.full_path(save_directory, create=True)

# Load and download the dataset, and save it to the specified folder
dataset = load_dataset("OptimalScale/ClimbMix", cache_dir=save_directory)

# Print the dataset structure
print(dataset)

