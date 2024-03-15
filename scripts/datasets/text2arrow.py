import os
from pathlib import Path
import datasets
from datasets import load_from_disk


def create_dataset_from_md_files(input_dir: str, output_dir: str):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get the list of .md file paths in the input directory
    md_file_paths = list(Path(input_dir).rglob("*.md"))
    print(f"Found {len(md_file_paths)} .md files in {input_dir}")

    # Define a generator function to yield examples
    def generate_examples():
        for file_path in md_file_paths:
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
                relative_path = str(file_path.relative_to(input_dir))
                yield {"text": content, "path": relative_path}

    # Create a HuggingFace dataset using the generator function
    dataset = datasets.Dataset.from_generator(generate_examples)

    # Save the dataset to the output directory
    dataset.save_to_disk(output_dir)

    print(f"Dataset created successfully in {output_dir}")


def test_saved_dataset(dataset_dir: str):
    # Load the saved dataset from disk
    dataset = load_from_disk(dataset_dir)

    # Get the number of items in the dataset
    num_items = len(dataset)
    print(f"Number of items in the dataset: {num_items}")

    # Print the first 5 examples
    print("First 5 examples:")
    for i in range(5):
        if i < num_items:
            example = dataset[i]
            print(f"Example {i + 1}:")
            print(example)
            print()
        else:
            break


if __name__ == "__main__":
    input_dir = r"D:\datasets\OpenTextBooks\markdown"
    output_dir = r"D:\datasets\OpenTextBooks\arrow"
    create_dataset_from_md_files(input_dir, output_dir)

    test_saved_dataset(output_dir)