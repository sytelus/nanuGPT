from typing import List, Union

import torch

TokenType = Union[int, str]

DIVISION_MODULO_OPERATIONS = {
    "x/y": lambda x, y, p: (x*y % p, y, x), # out = a,b,c such that b*c (mod p) = a (remember to detokenize!)
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

class GrokkingTokenizer:
    def __init__(self, prime: int, operations: list[str],
                 eos_token="<|eos|>", eq_token="="):
        self.prime = prime
        self.operations = operations
        self.eos_token = eos_token
        self.eq_token = eq_token

        # all_tokens is mix of numbers and strings
        self.all_tokens = [self.eos_token, self.eq_token] +  \
                    list(sorted(operations)) + \
                    list(range(prime))
        self.vocab_size = len(self.all_tokens)
        self.token_to_idx = {token: idx for idx, token in enumerate(self.all_tokens)}
        self.idx_to_token = {idx: token for idx, token in enumerate(self.all_tokens)}

    def __len__(self):
        return self.vocab_size

    def __getitem__(self, token: TokenType)->int:
        return self.token_to_idx[token]

    def __call__(self, token: TokenType)->int:
        return self.token_to_idx[token]

    def decode(self, idx: int)->TokenType:
        return self.idx_to_token[idx]

    def encode(self, token: TokenType)->int:
        return self.token_to_idx[token]

    def encode_batch(self, tokens: list[TokenType])->list[int]:
        return [self.encode(token) for token in tokens]

    def decode_batch(self, idxs: list[int])->list[TokenType]:
        return [self.decode(idx) for idx in idxs]

    # this method can only be used to encode tensor of numbers
    def encode_tensor(self, tokens: torch.Tensor)->torch.Tensor:
        return torch.tensor(self.encode_batch(tokens.tolist()))
    # this method can only be used to decode tensor of numbers
    def decode_tensor(self, idxs: torch.Tensor)->torch.Tensor:
        return torch.tensor(self.decode_batch(idxs.tolist()))

def get_tokenizer(prime:int):
    tokenizer = GrokkingTokenizer(prime, list(ALL_OPERATIONS.keys()))
    return tokenizer