from typing import Tuple
from math import ceil
import torch
from torch.utils.data import DataLoader

from grokking.arithmatic_tokenizer import ArithmaticTokenizer


DIVISION_MODULO_OPERATIONS = {
    "x/y": lambda x, y, p: (x*y % p, y, x), # out = a,b,c such that a/b (mod p) = c (mod p)
    "(x//y)if(y%2==1)else(x-y)": lambda x, y, _: torch.where(y % 2 == 1, x // y, x - y)
}

ALL_MODULO_OPERATIONS = {
    "x+y": lambda x, y, _: (x, y, x + y),
    "x-y": lambda x, y, _: (x, y, x - y),
    **DIVISION_MODULO_OPERATIONS,
    "x^2+y^2": lambda x, y, _: (x, y, x**2 + y**2),
    "x^2+xy+y^2": lambda x, y, _: (x, y, x**2 + x*y + y**2),
    "x^2+xy+y^2+x": lambda x, y, _: (x, y, x**2 + x*y + y**2 + x),
    "x^3+xy": lambda x, y, _: (x, y, x**3 + x*y),
    "x^3+xy^2+x": lambda x, y, _: (x, y, x**3 + x*y**2 + y)
}

ALL_OPERATIONS = {
    **ALL_MODULO_OPERATIONS,
}

def operation_mod_p_data(operation: str, p: int, tokenizer: ArithmaticTokenizer):
    """
    x◦y (mod p) for 0 <= x < p, 1 <= y < p if operation in DIVISION_MODULO_OPERATIONS
    x◦y (mod p) for 0 <= x, y < p otherwise
    """
    x = torch.arange(0, p)
    # for division operations, y starts at 1 to avoid divide by 0
    y = torch.arange(0 if not operation in DIVISION_MODULO_OPERATIONS else 1, p)
    x, y = torch.cartesian_prod(x, y).T

    eos = torch.ones_like(x) * tokenizer[tokenizer.eos_token]
    eq = torch.ones_like(x) * tokenizer[tokenizer.eq_token]
    op = torch.ones_like(x) * tokenizer[operation]

    x, y, z = ALL_OPERATIONS[operation](x, y, p)
    results = z.remainder(p)

    inputs = torch.stack([eos, tokenizer.encode_tensor(x), op, tokenizer.encode_tensor(y), eq], dim=1)
    labels = tokenizer.encode_tensor(results)

    return inputs, labels

def get_data(operation: str, prime: int, training_fraction: float,
             batch_size: int)->Tuple[DataLoader,DataLoader, ArithmaticTokenizer]:
    tokenizer = ArithmaticTokenizer(prime, list(ALL_OPERATIONS.keys()))

    inputs, labels = operation_mod_p_data(operation, prime, tokenizer)
    dataset = torch.utils.data.TensorDataset(inputs, labels)

    train_size = int(training_fraction * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    batch_size = min(batch_size, ceil(len(dataset) / 2))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, tokenizer
