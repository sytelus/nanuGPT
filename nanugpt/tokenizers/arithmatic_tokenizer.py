from typing import List, Union, Optional, Callable, Mapping

import numpy as np
import torch

from nanugpt.tokenizers.tokenizer_base import TokenizerBase


class ArithmaticTokenizer(TokenizerBase):
    def __init__(self, bos_str="$", pad_str="_", eos_str="|"):
        self.bos_str = bos_str
        self.pad_str = pad_str
        self.eos_str = eos_str

        tokens = (
            [str(d) for d in range(10)] +
            [self.bos_str, self.pad_str, self.eos_str] +
            ['=', '.', 'E', 'e', 'i', '+', '-', '*', '/', '^', '%', '\\', '~', ' ', '\n', '\r', '<', '>', ','] #'π', '∞', '∀', '…', '❌'
        )
        self.idx_to_token = {i: token for i, token in enumerate(tokens)}
        assert all(len(token) == 1 for token in self.idx_to_token.values()), "All tokens should have length 1"
        # we use 255 as not found token
        assert len(self.idx_to_token) < 255, "Number of tokens should be less than 255"

        self.vocab_size = len(self.idx_to_token)
        self.token_to_idx = {token: i for i, token in self.idx_to_token.items()}
        self.eos_token_id = self.token_to_idx[self.eos_str]

        # below are used in config
        assert self.eos_token_id == 12, "The eos token should be at index 12"
        assert self.token_to_idx['='] == 13, "The eq token should be at index 13"
        assert  self.token_to_idx[self.pad_str] == 11, "The pad token should be at index 11"

        # Cache lookup tables to avoid recomputation in tensor2strings and strings2tensor
        self._lookup = [self.idx_to_token[i] for i in range(self.vocab_size)]
        self._lut = np.full(256, 255, dtype=np.uint16)
        for token, index in self.token_to_idx.items():
            self._lut[ord(token)] = index

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

    def get_name(self)->str:
        return f'arithmatic_tokenizer'

    def tensor2strings(self, tensor: torch.Tensor) -> List[str]:
        tokens_np = tensor.cpu().numpy()
        return [''.join(self._lookup[token] for token in row) for row in tokens_np]

    def strings2tensor(self, strings: List[str]) -> torch.Tensor:
        if not strings:
            return torch.empty(0, dtype=torch.int16)
        n, l = len(strings), len(strings[0])
        big_str = ''.join(strings)
        arr = np.frombuffer(big_str.encode('latin1'), dtype=np.uint8).reshape(n, l)
        tokens = self._lut[arr]
        # assert none of the tokens are 255
        if tokens.max() == 255:
            raise AssertionError("Invalid chars exist in input")
        return torch.from_numpy(tokens)

def get_tokenizer_factory()->Callable[[], ArithmaticTokenizer]:
    return lambda : ArithmaticTokenizer()
