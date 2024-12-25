from typing import Any, Dict

import torch

from nanugpt.scalers.scaler_base import ScalerBase
from nanugpt.utils import TorchInfo

class KellerScaler(ScalerBase):
    def __init__(self, torch_info: TorchInfo):
        pass

    def backward(self, loss):
        # backward pass, with gradient scaling if training in fp16
        loss.backward()

    # clip the gradients and returns the pre-clip norm
    def clip(self, model, optimizer, grad_clip:float)->float:
        pre_clip_norm = -1.0 # pre-clip norm not available
        if grad_clip != 0.0:
            for p in model.parameters():
                p.grad = grad_clip*p.grad / (p.grad.norm() + 1e-6)
        return pre_clip_norm

    def step(self, optimizer):
        # step the optimizer (if grad were unscaled then scaler remembers and doesn't unscale again)
        optimizer.step()

    def update(self):
        pass

    def state_dict(self)->Dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        assert len(state_dict) == 0, "KellerScaler does not have any state to load"
        pass



def get_scaler(torch_info: TorchInfo)->ScalerBase:
    return KellerScaler(torch_info)