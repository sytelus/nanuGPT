

from typing import Any, Dict
from abc import ABC, abstractmethod
from nanugpt.utils import TorchInfo

class ScalerBase(ABC):
    @abstractmethod
    def backward(self, loss):
        pass
    @abstractmethod
    # clip the gradients and returns the pre-clip norm
    def clip(self, model, optimizer, grad_clip:float)->float:
        # pre-clip norm not available if -1.0
        pass
    @abstractmethod
    def step(self, optimizer):
        pass
    @abstractmethod
    def update(self):
        pass
    @abstractmethod
    def state_dict(self)->Dict[str, Any]:
        pass
    @abstractmethod
    def load_state_dict(self, state_dict: Dict[str, Any]):
        pass
