from typing import List, Mapping, Optional, Callable

import tiktoken

from nanugpt.tokenizers.tokenizer_base import TokenizerBase

class TiktokenWrap(TokenizerBase):
    def __init__(self, encoding_name:str):
        self.encoding_name = encoding_name
        self.tokenizer = tiktoken.get_encoding(encoding_name)

    def batch_encode(self, texts:List[str])->Mapping:
        return {'input_ids': self.tokenizer.encode_ordinary_batch(texts)}
        # EOT is appended in tokenize_dataset.py

    def batch_decode(self, ids:List[List[int]])->List[str]:
        return self.tokenizer.decode_batch(ids)

    def eot_token_id(self)->Optional[int]:
        return self.tokenizer.eot_token

    def get_name(self)->str:
        return f'tiktoken_{self.encoding_name}'

    def __len__(self):
        return self.tokenizer.max_token_value+1


def get_tokenizer_factory(encoding_name:str)->Callable[[], TokenizerBase]:
    return lambda : TiktokenWrap(encoding_name)
