from typing import List, Union

import torch

TokenType = Union[int, str]

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