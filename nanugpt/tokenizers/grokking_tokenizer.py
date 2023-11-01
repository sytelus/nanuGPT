from typing import List, Union, Optional, Callable, Mapping

import torch

from nanugpt.tokenizers.tokenizer_base import TokenizerBase


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

class GrokkingTokenizer(TokenizerBase):
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

        self.eos_token_id = self.token_to_idx[self.eos_token]

    def __len__(self):
        return self.vocab_size

    def __getitem__(self, token: TokenType)->int:
        return self.token_to_idx[token]

    def __call__(self, token: TokenType)->int:
        return self.token_to_idx[token]

    def decode(self, idx: int)->TokenType:
        return self.idx_to_token[idx]

    def encode(self, token: TokenType)->int:
        """Token is number of chars like =, +, -, x, y, etc."""
        return self.token_to_idx[token]

    def batch_encode(self, batch: List[List[TokenType]])->Mapping:
        return {'input_ids':
                    [[self.encode(item) for item in row]+[self.eos_token_id] for row in batch]
        }

    def batch_decode(self, batch: List[List[int]])->List[List[TokenType]]:
        return [[self.decode(idx) for idx in row] for row in batch]

    def eot_token_id(self)->Optional[int]:
        return self.eos_token_id

    def get_name(self)->str:
        return f'grokking_tokenizer_{self.prime}'

    # this method can only be used to encode tensor of numbers
    def encode_tensor(self, tokens: torch.Tensor)->torch.Tensor:
        return torch.tensor(self.batch_encode(tokens.tolist()))
    # this method can only be used to decode tensor of numbers
    def decode_tensor(self, idxs: torch.Tensor)->torch.Tensor:
        return torch.tensor(self.batch_decode(idxs.tolist()))

def get_tokenizer_factory(prime:int)->Callable[[], GrokkingTokenizer]:
    return lambda : GrokkingTokenizer(prime, list(ALL_OPERATIONS.keys()))
