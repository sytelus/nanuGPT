from typing import Tuple

import torch

def get_loss(model_output, labels)->Tuple[torch.Tensor, torch.Tensor]:
    # model_output: [batch_size, seq_len, vocab_size]
    # cross entropy loss expects a tensor of shape [batch_size, num_classes] and [batch_size]

    preds = model_output.view(-1, model_output.size(-1)) # [batch_size*seq_len, vocab_size]
    targets = labels.view(-1) # [batch_size*seq_len]

    # ignore_index=-1 is actually not needed because we never output -ve index for tokens.
    # PyTorch default is -100. The negative index is used to ignore the loss for padding tokens.
    loss = torch.nn.functional.cross_entropy(preds, targets, ignore_index=-1)
    correct = (torch.argmax(preds, dim=-1) == labels).sum()

    return loss, correct

