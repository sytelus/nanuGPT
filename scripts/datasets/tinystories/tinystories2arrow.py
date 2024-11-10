import os
from pathlib import Path
from typing import Iterator
import datasets
from datasets import load_from_disk, DatasetDict
import nanugpt.utils as utils

def read_delimited_parts(input_filepath: str, delimiter_str: str, strip: bool = False) -> Iterator[dict]:
    """
    Generator function that reads a UTF-8 encoded text file and yields parts delimited by delimiter_str.

    Args:
        input_filepath (str): Path to the input text file
        delimiter_str (str): String delimiter to split on
        strip (bool): Whether to strip whitespace from parts before yielding

    Yields:
        str: Each delimited part from the file
    """
    input_filepath = utils.full_path(input_filepath)
    with open(input_filepath, 'r', encoding='utf-8') as file:
        content = file.read()

    # Split content on delimiter
    parts = content.split(delimiter_str)

    # Yield each part, optionally stripped
    for part in parts:
        yield {"text": part.strip() if strip else part}

def file2arrow(input_filepath: str)->datasets.Dataset:
    input_filepath = utils.full_path(input_filepath)

    # Define a generator function to yield examples
    def generate_examples():
        return read_delimited_parts(input_filepath, delimiter_str=r"<|endoftext|>", strip=True)

    # Create a HuggingFace dataset using the generator function
    dataset = datasets.Dataset.from_generator(generate_examples)

    return dataset # type: ignore


def test_saved_dataset(dataset_dir: str):
    dataset_dir = utils.full_path(dataset_dir)
    print(f"Loading dataset from {dataset_dir}")

    # Load the saved dataset from disk
    datasets = load_from_disk(dataset_dir)

    for split, dataset in datasets.items(): # type: ignore
        # Get the number of items in the dataset
        num_items = len(dataset)
        print(f"Number of items in {split}: {num_items}")

        # Print the first 5 examples
        print("First 2 examples:")
        for i in range(2):
            if i < num_items:
                example = dataset[i]
                print(f"Example {i + 1}:")
                print(example)
                print()
            else:
                break


if __name__ == "__main__":
    train_ds = file2arrow("$DATA_ROOT/datasets/tinystories/tinystories_v2/TinyStoriesV2-GPT4-train.txt")
    validation_ds = file2arrow("$DATA_ROOT/datasets/tinystories/tinystories_v2/TinyStoriesV2-GPT4-valid.txt")

    output_dir = utils.full_path("$DATA_ROOT/datasets/tinystories/tinystories_v2")
    os.makedirs(output_dir, exist_ok=True)
    combined = DatasetDict({"train": train_ds, "validation": validation_ds})
    combined.save_to_disk(output_dir)

    test_saved_dataset(output_dir)
