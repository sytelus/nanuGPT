from typing import Any, Dict

import torch

from nanugpt.scalers.scaler_base import ScalerBase
from nanugpt.utils import TorchInfo

class AmpGradScaler(ScalerBase):
    def __init__(self, torch_info: TorchInfo):
        # initialize a GradScaler. If enabled=False scaler is a no-op
        # we need loss scaling only for fp16 due to reduced precision, not bf16 or fp32
        self.scaler = torch.amp.GradScaler("cuda", enabled=(torch_info.pt_dtype == torch.float16)) # type: ignore

    def backward(self, loss):
        # backward pass, with gradient scaling if training in fp16
        self.scaler.scale(loss).backward()

    # clip the gradients and returns the pre-clip norm
    def clip(self, model, optimizer, grad_clip:float)->float:
        pre_clip_norm = -1.0 # pre-clip norm not available
        if grad_clip != 0.0:
            # unscale the gradients and then clip
            self.scaler.unscale_(optimizer)
            pre_clip_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip).item()
        return pre_clip_norm

    def step(self, optimizer):
        # step the optimizer (if grad were unscaled then scaler remembers and doesn't unscale again)
        self.scaler.step(optimizer)

    def update(self):
        # update the scale for next iteration
        self.scaler.update()

    def state_dict(self)->Dict[str, Any]:
        return self.scaler.state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.scaler.load_state_dict(state_dict)



def get_scaler(torch_info: TorchInfo)->ScalerBase:
    return AmpGradScaler(torch_info)