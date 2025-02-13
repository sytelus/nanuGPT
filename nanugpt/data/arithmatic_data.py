import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
import numpy as np
import concurrent.futures
from functools import partial

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from nanugpt import utils
from nanugpt.tokenizers.arithmatic_tokenizer import get_tokenizer_factory, ArithmaticTokenizer

def _pack_sequences(docs: List[str], context_length: int, pad: str) -> List[str]:
    sequences = []
    current_seq = ""
    for doc in docs:
        if len(current_seq) + len(doc) <= context_length:
            current_seq += doc
        else:
            needed = context_length - len(current_seq)
            pad_str = (pad * ((needed // len(pad)) + 1))[:needed]
            sequences.append(current_seq + pad_str)
            current_seq = doc
        if len(current_seq) == context_length:
            sequences.append(current_seq)
            current_seq = ""
    if current_seq:
        needed = context_length - len(current_seq)
        pad_str = (pad * ((needed // len(pad)) + 1))[:needed]
        sequences.append(current_seq + pad_str)
    return sequences

def _process_n_digits(n_digits: int, max_samples: int, context_length: int,
                      op: str, eq: str, sep: str, pad: str,
                      seed:int) -> Tuple[List[str], List[str]]:
    low = 10**(n_digits - 1)
    high = 10**n_digits - 1
    total_count = (high - low + 1) ** 2

    if total_count <= max_samples:
        numbers = np.arange(low, high + 1)
        x = np.repeat(numbers, numbers.size)
        y = np.tile(numbers, numbers.size)
    else:
        rng = np.random.default_rng(seed)
        x = rng.integers(low, high + 1, size=max_samples)
        y = rng.integers(low, high + 1, size=max_samples)

    z = x * y
    x_str = np.char.mod('%d', x)
    y_str = np.char.mod('%d', y)
    z_str = np.char.mod('%d', z)

    docs = np.char.add(np.char.add(np.char.add(np.char.add(x_str, op), y_str), eq), z_str)
    docs = np.char.add(docs, sep)
    docs_list:List[str] = docs.tolist() # type: ignore
    sequences = _pack_sequences(docs_list, context_length, pad)

    if total_count <= max_samples:
        return sequences, []
    else:
        return sequences[::2], sequences[1::2]

def gen_mul_seqs(min_digits: int, max_digits: int, max_samples: int, context_length: int,
                 op: str, eq: str, sep: str, pad: str, seed:int) -> Tuple[List[str], List[str]]:
    train_sequences = []
    test_sequences = []

    # Create a partial function that fixes the parameters except n_digits.
    process_func = partial(_process_n_digits, max_samples=max_samples,
                           context_length=context_length, op=op, eq=eq, sep=sep, pad=pad,
                           seed=seed)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(process_func, range(min_digits, max_digits + 1))
        for train_seq, test_seq in results:
            train_sequences.extend(train_seq)
            test_sequences.extend(test_seq)

    return train_sequences, test_sequences

def shift_by_1(tensor: torch.Tensor, eos_token_id: int) -> torch.Tensor:
    shifted = torch.empty_like(tensor)
    shifted[:, :-1] = tensor[:, 1:]
    shifted[:, -1] = eos_token_id
    return shifted

def get_data(device_batch_size: int, eval_batch_size:int, data_loader_seed:int,
             min_digits: int, max_digits: int, max_samples: int, context_length: int)->Tuple[DataLoader,DataLoader, Optional[DataLoader]]:

    world_size = utils.get_world_size()
    global_rank = utils.get_global_rank()
    local_world_size = utils.get_local_world_size()
    assert world_size > 0 and global_rank >= 0 and global_rank < world_size and local_world_size >= 1, f"Invalid values: Word size={world_size}, global rank={global_rank}, local world size={local_world_size}"

    tokenizer:ArithmaticTokenizer = get_tokenizer_factory()()

    seed = data_loader_seed + global_rank

    train_strings, val_strings = gen_mul_seqs(min_digits, max_digits, max_samples, context_length,
                           op='*', eq='=',
                           sep=tokenizer.eos_str, pad=tokenizer.pad_str,
                           seed=seed)
    train_tensors = tokenizer.strings2tensor(train_strings).to(torch.long)
    val_tensors = tokenizer.strings2tensor(val_strings).to(torch.long)

    train_dataset = torch.utils.data.TensorDataset(train_tensors, shift_by_1(train_tensors, tokenizer.eos_token_id))
    val_dataset = torch.utils.data.TensorDataset(val_tensors, shift_by_1(val_tensors, tokenizer.eos_token_id))

    train_loader_seed, val_loader_seed, test_loader_seed = data_loader_seed+global_rank, data_loader_seed+global_rank + 1, data_loader_seed+global_rank + 2
    train_loader_gen = torch.Generator().manual_seed(train_loader_seed)
    val_loader_gen = torch.Generator().manual_seed(val_loader_seed)

    train_loader = DataLoader(train_dataset,
                              batch_size=min(device_batch_size, len(train_dataset)) ,
                              shuffle=False,
                              generator=train_loader_gen)
    val_loader = DataLoader(val_dataset,
                            batch_size=min(eval_batch_size, len(val_dataset)) ,
                            shuffle=False,
                            generator=val_loader_gen)

    test_loader = None

    return train_loader, val_loader, test_loader
