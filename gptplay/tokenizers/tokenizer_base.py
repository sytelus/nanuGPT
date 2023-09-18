from abc import abstractmethod
from typing import List, Mapping, Optional

class TokenizerBase:
    @abstractmethod
    def batch_encode(self, texts:List[str])->Mapping:
        raise NotImplementedError

    @abstractmethod
    def batch_decode(self, ids:List[List[int]])->List[str]:
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

