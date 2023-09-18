import os

from gptplay.data import tokenize_hf_dataset
from gptplay import utils

if __name__ == '__main__':
    tokenized_out_dir = os.environ.get('DATA_ROOT', os.path.dirname(__file__))
    tokenized_out_dir = utils.full_path(os.path.join(tokenized_out_dir, 'tokenized'), create=True)

    dataset_save_dir = os.environ.get('DATA_ROOT', os.path.dirname(__file__))

    tokenize_hf_dataset.tokenize(dataset_path="roneneldan/TinyStories",
                            dataset_name="tinystories_v1",
                            tokenized_out_dir=tokenized_out_dir,
                            data_files={
                                "train": "TinyStories-train.txt",
                                "validation": "TinyStories-valid.txt",
                            },
                            dataset_save_dir=utils.full_path(os.path.join(dataset_save_dir, 'tinystories_v1'), create=True),
                            )

    tokenize_hf_dataset.tokenize(dataset_path="roneneldan/TinyStories",
                            dataset_name="tinystories_v2",
                            tokenized_out_dir=tokenized_out_dir,
                            data_files={
                                "train": "TinyStoriesV2-GPT4-train.txt",
                                "validation": "TinyStoriesV2-GPT4-valid.txt",
                            },
                            dataset_save_dir=utils.full_path(os.path.join(dataset_save_dir, 'tinystories_v2'), create=True),
                            )