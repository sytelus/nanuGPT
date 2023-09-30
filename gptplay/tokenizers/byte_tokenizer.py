from typing import List, Mapping, Optional, Callable

import numpy as np

from gptplay.tokenizers.tokenizer_base import TokenizerBase

class ByteTokenizer(TokenizerBase):
    def __init__(self, encoding_name='utf-8', append_eot=True):
        self.encoding_name = encoding_name
        self.special_tokens = {'<EOS>': 256, '<PAD>': 257, '<UNK>': 258, '<BOS>': 259}
        self.reverse_special_tokens = {v: k for k, v in self.special_tokens.items()}
        self.eot_token = '<EOS>'
        self.append_eot = append_eot
        self._eot_token_id = self.special_tokens[self.eot_token]

    def encode(self, text:str)->List[int]:
        encoded = []
        i = 0
        while i < len(text):
            if len(self.special_tokens):
                found_tokens = [(text.find(token, i), token) for token in self.special_tokens.keys()]
                found_tokens = [(k,t) for k,t in found_tokens if k>= 0] # filter out not found
                if len(found_tokens) > 0:
                    first_special = min(found_tokens, key=lambda x: x[0])
                    encoded.extend(list(text[i:first_special[0]].encode(encoding=self.encoding_name)))
                    encoded.append(self.special_tokens[first_special[1]])
                    i = first_special[0] + len(first_special[1])
                    continue
                #else  no special tokens found
            #else  no special tokens found
            encoded.extend(list(text[i:].encode(encoding=self.encoding_name)))
            i = len(text)

        return encoded

    def decode(self, ids:List[int])->str:
        decoded = ""
        last_index = 0

        for i, id in enumerate(ids):
            if id in self.reverse_special_tokens:
                decoded += bytes(ids[last_index:i]).decode(encoding=self.encoding_name)
                decoded += self.reverse_special_tokens[id]
                last_index = i+1
        decoded += bytes(ids[last_index:len(ids)]).decode(encoding=self.encoding_name)
        return decoded

    def batch_encode(self, texts:List[str])->Mapping:
        return {'input_ids': [self.encode(text) for text in texts]}

    def batch_decode(self, ids_batch:List[List[int]])->List[str]:
        return [self.decode(ids) for ids in ids_batch]

    def eot_token_id(self)->Optional[int]:
        return self._eot_token_id

    def get_name(self)->str:
        return f'byte_tokenizer_{self.encoding_name}'

    def __len__(self):
        return 256 + len(self.special_tokens)


def get_tokenizer_factory(encoding_name:str)->Callable[[], TokenizerBase]:
    return lambda : ByteTokenizer(encoding_name)
