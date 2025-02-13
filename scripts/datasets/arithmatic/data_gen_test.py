import torch
import numpy as np
from typing import List, Dict
from typing import List, Tuple
import numpy as np
import concurrent.futures
from functools import partial

def _pack_sequences(docs: List[str], context_len: int, pad: str) -> List[str]:
    sequences = []
    current_seq = ""
    for doc in docs:
        if len(current_seq) + len(doc) <= context_len:
            current_seq += doc
        else:
            needed = context_len - len(current_seq)
            pad_str = (pad * ((needed // len(pad)) + 1))[:needed]
            sequences.append(current_seq + pad_str)
            current_seq = doc
        if len(current_seq) == context_len:
            sequences.append(current_seq)
            current_seq = ""
    if current_seq:
        needed = context_len - len(current_seq)
        pad_str = (pad * ((needed // len(pad)) + 1))[:needed]
        sequences.append(current_seq + pad_str)
    return sequences

def _process_n_digits(n_digits: int, max_samples: int, context_len: int, op: str, eq: str, sep: str, pad: str) -> Tuple[List[str], List[str]]:
    low = 10**(n_digits - 1)
    high = 10**n_digits - 1
    total_count = (high - low + 1) ** 2

    if total_count <= max_samples:
        numbers = np.arange(low, high + 1)
        x = np.repeat(numbers, numbers.size)
        y = np.tile(numbers, numbers.size)
    else:
        x = np.random.randint(low, high + 1, size=max_samples)
        y = np.random.randint(low, high + 1, size=max_samples)

    z = x * y
    x_str = np.char.mod('%d', x)
    y_str = np.char.mod('%d', y)
    z_str = np.char.mod('%d', z)

    docs = np.char.add(np.char.add(np.char.add(np.char.add(x_str, op), y_str), eq), z_str)
    docs = np.char.add(docs, sep)
    docs_list = docs.tolist()
    sequences = _pack_sequences(docs_list, context_len, pad)

    if total_count <= max_samples:
        return sequences, []
    else:
        return sequences[::2], sequences[1::2]

def gen_mul_seqs(min_digits: int, max_digits: int, max_samples: int, context_len: int,
                 op: str, eq: str, sep: str, pad: str) -> Tuple[List[str], List[str]]:
    train_sequences = []
    test_sequences = []

    # Create a partial function that fixes the parameters except n_digits.
    process_func = partial(_process_n_digits, max_samples=max_samples,
                           context_len=context_len, op=op, eq=eq, sep=sep, pad=pad)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(process_func, range(min_digits, max_digits + 1))
        for train_seq, test_seq in results:
            train_sequences.extend(train_seq)
            test_sequences.extend(test_seq)

    return train_sequences, test_sequences

def strs2tokens(strings: List[str], mapping: Dict[str, int]) -> torch.Tensor:
    if not strings:
        return torch.empty(0, dtype=torch.int16)
    n, l = len(strings), len(strings[0])
    # Build lookup table for 256 possible byte values (assumes mapping keys are single-byte)
    lut = np.zeros(256, dtype=np.uint16)
    for ch, token in mapping.items():
        lut[ord(ch)] = token
    # Join all strings and encode using 'latin1' (1:1 mapping for bytes)
    big_str = ''.join(strings)
    arr = np.frombuffer(big_str.encode('latin1'), dtype=np.uint8).reshape(n, l)
    tokens = lut[arr]
    return torch.from_numpy(tokens)

# Example usage:
if __name__ == '__main__':
    train, test = gen_mul_seqs(
        min_digits=1,
        max_digits=2,
        max_samples=10,
        context_len=15,
        op='*',
        eq='=',
        sep='|',
        pad='_'
    )
    print("Train sequences:")
    for seq in train:
        print(seq)
    print("\nTest sequences:")
    for seq in test:
        print(seq)

    # Tokenize the sequences
    mapping = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '*': 10, '=': 11, '|': 12, '_': 13}
    tokens = strs2tokens(train + test, mapping)
    print("\nTokens:")
    print(tokens)