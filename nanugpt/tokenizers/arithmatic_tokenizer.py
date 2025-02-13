from typing import List, Union, Optional, Callable, Mapping

import numpy as np
import torch

from nanugpt.tokenizers.tokenizer_base import TokenizerBase


class ArithmaticTokenizer(TokenizerBase):
    def __init__(self, bos_token="$", pad_token="_", eos_token="|"):
        self.bos_token = bos_token
        self.pad_token= pad_token
        self.eos_token = eos_token

        self.idx_to_token = {
            i: val
            for i, val in enumerate( \
                list(str(d) for d in range(10)) +
                [self.bos_token, self.pad_token, self.eos_token] + \
                ['.', 'E', 'e', 'i', '+', '-', '*', '/', '^', '%', '\\', '~', ' ', '\n', '\r', 'π', '∞', '<', '>', ',', '∀', '…', '❌']
            )
        }
        # assert that length of each token is 1
        assert all(len(token) == 1 for token in self.idx_to_token.values()), "All tokens should have length 1"

        self.vocab_size = len(self.idx_to_token)
        self.token_to_idx = {val: i for i, val in self.idx_to_token.items()}

        self.eos_token_id = self.token_to_idx[self.eos_token]

    def __len__(self):
        return self.vocab_size

    def __getitem__(self, token: str)->int:
        return self.token_to_idx[token]

    def __call__(self, token: str)->int:
        return self.token_to_idx[token]

    def decode(self, idx: int)->str:
        return self.idx_to_token[idx]

    def encode(self, token: str)->int:
        return self.token_to_idx[token]

    def batch_encode(self, batch: List[List[str]]) -> Mapping:
        token_to_idx = self.token_to_idx
        eos = self.eos_token_id
        return {'input_ids': [ [token_to_idx[token] for token in row] + [eos] for row in batch ]}

    def batch_decode(self, batch: List[List[int]]) -> List[List[str]]:
        idx_to_token = self.idx_to_token
        return [[idx_to_token[idx] for idx in row] for row in batch]

    def tensor2strings(self, tensor: torch.Tensor) -> List[str]:
        tokens_np = tensor.cpu().numpy()
        lookup = [self.idx_to_token[i] for i in range(self.vocab_size)]
        return [''.join(lookup[token] for token in row) for row in tokens_np]

    def get_name(self)->str:
        return f'arithmatic_tokenizer'

    def strings2tensor(self, strings: List[str]) -> torch.Tensor:
        if not strings:
            return torch.empty(0, dtype=torch.int16)
        n, l = len(strings), len(strings[0])
        # Build lookup table for 256 possible byte values (assumes mapping keys are single-byte)
        lut = np.zeros(256, dtype=np.uint16)
        for ch, token in self.token_to_idx.items():
            lut[ord(ch)] = token
        # Join all strings and encode using 'latin1' (1:1 mapping for bytes)
        big_str = ''.join(strings)
        arr = np.frombuffer(big_str.encode('latin1'), dtype=np.uint8).reshape(n, l)
        tokens = lut[arr]
        return torch.from_numpy(tokens)


def get_tokenizer_factory(prime:int)->Callable[[], ArithmaticTokenizer]:
    return lambda : ArithmaticTokenizer()
