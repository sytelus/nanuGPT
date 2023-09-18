from abc import abstractmethod
from typing import List

class TokenizerBase:
    @abstractmethod
    def batch_encode(self, texts:List[str])->Mapping:
        raise NotImplementedError

    @abstractmethod
    def batch_decode(self, ids:List[List[int]])->List[str]:
        raise NotImplementedError

