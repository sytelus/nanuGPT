from abc import abstractmethod
from typing import List, Mapping, Optional

"""
Defines the interface for al tokenizers.
"""
class TokenizerBase:
    @abstractmethod
    def batch_encode(self, batch:list)->Mapping:
        raise NotImplementedError

    @abstractmethod
    def batch_decode(self, batch:list)->list:
        raise NotImplementedError

    @abstractmethod
    def eot_token_id(self)->Optional[int]:
        raise NotImplementedError

    @abstractmethod
    def get_name(self)->str:
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

