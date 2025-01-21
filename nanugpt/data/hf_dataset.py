from typing import Optional, Tuple, List, Dict, Mapping, Callable, MutableMapping
import os

from datasets import DatasetDict, load_dataset, load_from_disk

import torch
from torch.utils.data import DataLoader

from nanugpt.tokenizers.tokenizer_base import TokenizerBase
from nanugpt import utils
from nanugpt import glogging as logging

"""
Module implementing `get_data` interface for HuggingFace datasets. Appropriate val and test splits are created if necessary. The dataloader is setup with on-the-fly tokenization.

Additional `get_datasets` function is provided to return the raw dataset.

This module allows to use most HugingFace text datasets without requiring pre-tokenization.
"""

def get_datasets(hf_name_path:str, hf_dataset_name:Optional[str], hf_data_dir:Optional[str], hf_data_files:Optional[str], hf_revision:Optional[str],
             train_split:Optional[str], val_split:Optional[str], test_split:Optional[str], hf_cache_dir:Optional[str],
             hf_sample_by:Optional[str],
             val_fraction:Optional[float], test_fraction:Optional[float],
             data_loader_seed:int)->Tuple[DatasetDict, Optional[str], Optional[str], Optional[str]] :
    """
    Load HuggingFace dataset. The huggingface dataset is specified by dataset name and can be further filtered by specifying the subset name (which could be specific cawls, smaller samples etc) and/or specific directory and/or specific files. This is done by `load_dataset` function.

    For locally saved HF datasets, we use `load_from_disk`. Note that these are cached datasets which should use `load_dataset` which will hit the cache automatically. Instead these are usually the datasets created by `save_to_disk` which creates arrow files.

    Another type of local datasets is bunch of text, csv, json etc files. The can by loaed by `load_dataset` but with `path="text"` etc. In this case, the `hf_data_files` should be a dictionary with keys as splits and values as list of file paths.

    Parameters:
    hf_name_path (str): Local path or name of the HuggingFace dataset (example: "HuggingFaceFW/fineweb").
    hf_dataset_name (Optional[str]): Subset name to limit to (example: "sample-10BT").
    hf_data_dir (Optional[str]): Directory to limit the search for data files that is in "Files and versions" tab (example: "sample/10BT").
    hf_data_files (Optional[str]): Data files to limit to (example: "012_00000.parquet")
    hf_revision (Optional[str]): Dataset revision to use.
    train_split (Optional[str]): Name of the training split.
    val_split (Optional[str]): Name of the validation split.
    test_split (Optional[str]): Name of the test split.
    hf_cache_dir (Optional[str]): Directory to cache the dataset.
    hf_sample_by (Optional[str]): Sampling method.
    val_fraction (Optional[float]): Fraction of the dataset to use for validation.
    test_fraction (Optional[float]): Fraction of the dataset to use for testing.
    data_loader_seed (int): Seed for data shuffling.

    Returns:
    Tuple[DatasetDict, Optional[str], Optional[str], Optional[str]]:
        A tuple containing the dataset dictionary and the names of the train, validation, and test splits.
    """

    # if name is valid local path, load from disk using load_from_disk
    if os.path.isdir(os.path.expanduser(os.path.expandvars(hf_name_path))):
        hf_name_path = utils.full_path(hf_name_path)
        logging.info(f'Loading dataset from disk {hf_name_path}...')
        dataset = load_from_disk(hf_name_path)
    else:
        # if we are trying to load text, csv, json etc from local disk, hf_name_path should be "text" etc
        # in this case, make sure local paths are expanded
        if '/' not in hf_name_path and isinstance(hf_data_files, MutableMapping):
            # dict keys are splits
            # HuggingFace doesn't like MutableMapping and must have dict
            hf_data_files = dict(hf_data_files) # type: ignore
            # expand paths
            for split, filepath in hf_data_files.items():
                hf_data_files[split] = [utils.full_path(f) for f in hf_data_files[split]]

            logging.info(f'Loading text file(s) from {hf_name_path}...')

        # now load using load_dataset
        logging.info(f'Loading HuggingFace dataset {hf_name_path}...')
        dataset = load_dataset(hf_name_path, name=hf_dataset_name, data_dir=hf_data_dir,
                               data_files=hf_data_files, revision=hf_revision,
                               cache_dir=hf_cache_dir, sample_by=hf_sample_by)

    # If HF API returns just one dataset, convert it to DatasetDict
    if not isinstance(dataset, DatasetDict):
        dataset = DatasetDict({train_split: dataset})

    # log size of each split
    logging.info(f'Loaded dataset {hf_name_path}')
    dataset_split_names = set(dataset.keys())
    for split in dataset_split_names:
        logging.summary({f'data/{split}_original_rows': len(dataset[split])})

    # Make sure we have train split
    if train_split is None: # auto-detect
        if 'train' in dataset:
            train_split = 'train'
            logging.info('train_split is None, using the name "train" for the split')
        else:
            raise ValueError(f'train_split is None and no "train" split found in dataset {hf_name_path}')
    if train_split not in dataset:
        raise ValueError(f'No "{train_split}" split found in dataset {hf_name_path}')

    val_fraction = val_fraction or 0.
    test_fraction = test_fraction or 0.

    # if dataset doesn't have val_split, we need to create it from train_split
    if val_split is None:
        # detect val split
        usual_val_split_names = {'validation', 'valid', 'dev', 'val'}
        found_val_split = None
        for split in usual_val_split_names:
            if split in dataset:
                found_val_split = split
                logging.info(f'val_split is None, using the name "{found_val_split}" for the split')
                break
        # if val frac is specified but if val split already exist then its mistake
        if val_fraction:
            if found_val_split:
                raise ValueError(f'val_fraction={val_fraction} > 0, but a possible val split "{found_val_split}" was detected in dataset {hf_name_path}.')
            else:
                # we need to create a val split
                val_split = (dataset_split_names.intersection(usual_val_split_names) or {'validation'}).pop()
        else:
            val_split = found_val_split

    if test_split is None:
        # detect test split
        usual_test_split_names = {'test'}
        found_test_split = None
        for split in usual_test_split_names:
            if split in dataset:
                found_test_split = split
                logging.info(f'test_split is None, using the name "{found_test_split}" for the split')
                break
        # if test frac is specified but if test split already exist then its mistake
        if test_fraction:
            if found_test_split:
                raise ValueError(f'test_fraction={test_fraction} > 0, but a possible test split "{found_test_split}" was detected in dataset {hf_name_path}.')
            else:
                # we need to create a test split
                test_split = (dataset_split_names.intersection(usual_test_split_names) or {'test'}).pop()
        else:
            test_split = found_test_split

    if test_fraction: # simplify code
        assert val_fraction > 0, 'test_fraction can only be used if val_fraction > 0'

    # create or get splits
    if val_split not in dataset and (val_fraction+test_fraction)>0.:
        splits = dataset[train_split].train_test_split(test_size=val_fraction+test_fraction, shuffle=True, seed=data_loader_seed)
        dataset[train_split] = splits['train']
        if val_fraction:
            if not test_fraction:
                # there is no test split
                dataset[val_split] = splits['test']
            else:
                # there is a test split
                splits = splits['test'].train_test_split(test_size=test_fraction/(val_fraction+test_fraction), shuffle=True, seed=data_loader_seed)
                dataset[val_split] = splits['train']
                dataset[test_split] = splits['test']
        else:
            # there is only a test split
            dataset[test_split] = splits['test']
    else:
        logging.info(f'Using existing val_split "{val_split}", ignoring val_fraction={val_fraction}')
        if test_split not in dataset and test_fraction:
            splits = dataset[train_split].train_test_split(test_size=test_fraction, shuffle=True, seed=data_loader_seed)
            dataset[train_split] = splits['train']
            dataset[test_split] = splits['test']
        else:
            logging.info(f'Using existing test_split "{test_split}", ignoring test_fraction={test_fraction}')

    assert train_split in dataset
    assert (val_split and val_split in dataset) or val_fraction == 0
    assert (test_split and test_split in dataset) or test_fraction == 0
    assert train_split != val_split and train_split != test_split and val_split != test_split

    for split in dataset.keys():
        logging.summary({f'data/{split}_split_rows': len(dataset[split])})

    return dataset, train_split, val_split, test_split


def get_data(hf_name_path:str, hf_dataset_name:Optional[str], hf_data_dir:Optional[str], hf_data_files:Optional[str],
             hf_revision:Optional[str], hf_sample_by:Optional[str],
             train_split:Optional[str], val_split:Optional[str], test_split:Optional[str], hf_cache_dir:Optional[str],
             train_fraction:Optional[float], val_fraction:Optional[float], test_fraction:Optional[float],
             device_batch_size: int, eval_batch_size:int, data_loader_seed:int, text_column:str,
             local_rank:int, context_length:int, tokenizer_factory:Callable[[], TokenizerBase])->Tuple[DataLoader,DataLoader, Optional[DataLoader]]:

    dataset = get_datasets(hf_name_path=hf_name_path, hf_dataset_name=hf_dataset_name, hf_data_dir=hf_data_dir, hf_data_files=hf_data_files, hf_revision=hf_revision,
                           train_split=train_split, val_split=val_split, test_split=test_split, hf_cache_dir=hf_cache_dir,
                           hf_sample_by=hf_sample_by, val_fraction=val_fraction, test_fraction=test_fraction,
                           data_loader_seed=data_loader_seed)

    # get datasets
    train_dataset = dataset[train_split]
    val_dataset = dataset[val_split] if val_split in dataset else None
    test_dataset = dataset[test_split] if test_split in dataset else None

    # set on-the-fly tokenization
    # we need 3 different instances due to threading issues
    if train_dataset is not None:
        train_tokenizer = tokenizer_factory()
        train_dataset.set_transform(lambda x: train_tokenizer.batch_encode(x[text_column]))
        train_loader_gen = torch.Generator().manual_seed(data_loader_seed)
        train_loader = DataLoader(train_dataset,
                                  batch_size=min(device_batch_size, len(train_dataset)),
                                  shuffle=True,
                                  num_workers=0, # don't use main process as worker
                                  generator=train_loader_gen)
    if val_dataset is not None:
        val_tokenizer = tokenizer_factory()
        val_dataset.set_transform(lambda x: val_tokenizer.batch_encode(x[text_column]))
        val_loader_gen = torch.Generator().manual_seed(data_loader_seed)
        val_loader = DataLoader(val_dataset,
                                batch_size=min(eval_batch_size, len(val_dataset)) ,
                                shuffle=False,
                                num_workers=0, # don't use main process as worker
                                generator=val_loader_gen)
    if test_dataset is not None:
        test_tokenizer = tokenizer_factory()
        test_dataset.set_transform(lambda x: test_tokenizer.batch_encode(x[text_column]))
        test_loader_gen = torch.Generator().manual_seed(data_loader_seed)
        test_loader = DataLoader(test_dataset,
                                 batch_size=min(eval_batch_size, len(test_dataset)) ,
                                 shuffle=False,
                                 num_workers=0, # don't use main process as worker
                                 generator=test_loader_gen)

    return train_loader, val_loader, test_loader
