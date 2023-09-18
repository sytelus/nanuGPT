from typing import List, Mapping, Optional, Callable

from transformers import AutoTokenizer

from gptplay.tokenizers.tokenizer_base import TokenizerBase

class HfTokenizer(TokenizerBase):
    def __init__(self, hf_path:str, cache_dir:str, fix_pad_token:bool,
                 padding:bool, trucate:bool, truncation_side:Optional[str], padding_side:Optional[str],
                model_max_length:int, skip_special_decoded_tokens:bool,
                clean_up_tokenization_spaces:bool, **kwargs):

        self.padding = padding
        self.trucate = trucate
        self.skip_special_decoded_tokens = skip_special_decoded_tokens
        self.clean_up_tokenization_spaces = clean_up_tokenization_spaces

        self.tokenizer = AutoTokenizer.from_pretrained(hf_path, cache_dir=cache_dir,
                                                padding_side=padding_side,
                                                truncation_side=truncation_side,
                                                model_max_length=model_max_length,
                                                **kwargs)

        # pad token is required for unequal sequences. If tokenizer is not trained with
        # specific pad toeken, we can use EOS as pas token. This is done by torch
        # automatically but it spits out warnings each time and pollutes the log.
        if fix_pad_token:
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def batch_encode(self, texts:List[str])->Mapping:
        return self.tokenizer(texts,
                                padding=self.padding,
                                truncation=self.trucate,
                                return_tensors='pt')

    def batch_decode(self, ids:List[List[int]])->List[str]:
        return self.tokenizer.batch_decode(ids,
                                           skip_special_tokens=self.skip_special_decoded_tokens,
                                           clean_up_tokenization_spaces=self.clean_up_tokenization_spaces)

    def eot_token_id(self)->Optional[int]:
        return self.tokenizer.eos_token_id


def get_tokenizer_factory(hf_path:str, cache_dir:str, fix_pad_token:bool,
                  padding:bool, trucate:bool, truncation_side:Optional[str], padding_side:Optional[str],
                  model_max_length:int, **kwargs)->Callable[[], TokenizerBase]:

    return lambda : HfTokenizer(hf_path, cache_dir, fix_pad_token, padding, trucate,
                       truncation_side, padding_side, model_max_length, **kwargs)
