from typing import Tuple, Union, Optional, Callable, Mapping

import torch
from torch import nn

from nanugpt.common import GetLossType

class ModelWithLoss(nn.Module):
    """
    Wrap usual model that returns logits and combine loss function with it.
    torch.compile will fuse loss function that makes it much faster (30% at 124M params).
    """
    def __init__(self, model: nn.Module, get_loss:Optional[GetLossType]):
        super().__init__()
        self.model = model
        self.get_loss = get_loss

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_logits: bool = True,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: # logits, loss, num_correct, num_labels
        logits = self.model(input_ids)
        if isinstance(logits, Mapping): # support HF model output as dict
            logits = logits['logits']

        loss:Optional[torch.Tensor] = None
        if labels is not None:
            assert self.get_loss is not None, "Loss function is not defined"
            loss, correct = self.get_loss(logits, labels)
            # keeping logits around may unnecessarily consume a lot of memory  (atleast 1GB for 124M params)
            return logits if return_logits else None, loss, correct

        return logits, None, None

