from typing import List, Mapping, Optional, Callable, Any

from transformers import AutoTokenizer

from nanugpt.tokenizers.tokenizer_base import TokenizerBase

class HfTokenizer(TokenizerBase):
    def __init__(self,
                 hf_path:str,
                 name:Optional[str],
                 cache_dir:Optional[str],
                 fix_pad_token:bool,
                 padding:Optional[Any],
                 trucate:Optional[Any],
                 truncation_side:Optional[str],
                 padding_side:Optional[str],
                 model_max_length:Optional[int],
                 skip_special_decoded_tokens:Optional[bool],
                 clean_up_tokenization_spaces:Optional[bool],
                 **kwargs):

        self.name = name or hf_path
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
            if self.tokenizer.pad_token is None and self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def batch_encode(self, texts:List[str])->Mapping:
        return self.tokenizer(texts,
                                padding=self.padding, # type: ignore
                                truncation=self.trucate, # type: ignore
                                return_tensors='pt')

    def batch_decode(self, ids:List[List[int]])->List[str]:
        return self.tokenizer.batch_decode(ids,
            skip_special_tokens=self.skip_special_decoded_tokens, # type: ignore
            clean_up_tokenization_spaces=self.clean_up_tokenization_spaces, # type: ignore
        )

    def eot_token_id(self)->Optional[int]:
        return self.tokenizer.eos_token_id

    def get_name(self)->str:
        return self.name

    def __len__(self):
        return len(self.tokenizer)


def get_tokenizer_factory(
        hf_path:str,
        model_max_length:Optional[int]=None,
        name:Optional[str]=None,
        cache_dir:Optional[str]=None,
        fix_pad_token=True, # if pad_token is not set, set it to eos_token
        padding:Optional[Any]=True, # padd for sequences of unequal length
        trucate:Optional[Any]=True, # trucate sequences of unequal length
        truncation_side:Optional[str]='left', # truncate from left if too long seq
        padding_side:Optional[str]='left', # pad on left if too short seq
        clean_up_tokenization_spaces:Optional[bool]=None,
        skip_special_decoded_tokens:Optional[bool]=None,
        **kwargs
    )->Callable[[], TokenizerBase]:

    return lambda : HfTokenizer(hf_path, name, cache_dir, fix_pad_token, padding, trucate,
                       truncation_side, padding_side, model_max_length,
                       clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                       skip_special_decoded_tokens=skip_special_decoded_tokens,
                       **kwargs)
