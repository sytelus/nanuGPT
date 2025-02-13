from typing import Tuple, Mapping, Callable

import torch

def get_loss(model_output, labels)->Tuple[torch.Tensor, torch.Tensor]:

    if isinstance(model_output, Mapping):
        model_output = model_output['logits']

    # model output is tensor [5,batch_size,prime+2]
    # [EOS a op b =] is input to model which is 5 tokens
    # output is [a op b = c] which is 5 tokens
    # we only take the last token of the output for loss
    last_logits = model_output[-1,:,:] # [batch_size, vocab_size],only for the last prediction
    loss = torch.nn.functional.cross_entropy(last_logits, labels)
    # argmax over vocab_size to get the index of the predicted token for each item in batch
    correct = (torch.argmax(last_logits, dim=-1) == labels).sum()

    return loss, correct

def get_loss_factory()->Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    return get_loss
