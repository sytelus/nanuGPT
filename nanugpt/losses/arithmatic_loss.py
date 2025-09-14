from nanugpt import utils
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Callable

_eq_token_id: int = -1
_pad_token_id: int = -1
_eos_token_id: int = -1

def get_loss(model_output: Tensor,
             labels: Tensor,
            ) -> Tuple[Tensor, int]:
    r"""
    Compute cross entropy loss (only on answer tokens) and count how many documents
    have all tokens predicted correctly.

    The model_output is of shape [batch, context_len, vocab_size] and labels is of shape [batch, context_len].
    In each sequence, documents are separated by eos_token_id and within each document there is exactly one eq_token_id.
    Only tokens between the eq_token_id and the following eos_token_id (excluding both) are used to compute the loss.
    A document is considered "correct" if every token in its answer segment (the selected tokens) is predicted
    correctly.

    Padding tokens (pad_token_id) are ignored.
    """
    global _eq_token_id, _pad_token_id, _eos_token_id

    # Ensure we work on the same device as the inputs.
    device = model_output.device

    # Create boolean masks for the special tokens.
    eq_mask  = (labels == _eq_token_id)
    eos_mask = (labels == _eos_token_id)
    pad_mask = (labels == _pad_token_id)

    # Compute cumulative sums along the context (sequence) dimension.
    # These will help us decide if a token is in an answer segment.
    # For each position, eq_cumsum counts how many eq tokens have been seen so far.
    # Similarly, eos_cumsum counts how many eos tokens have been seen.
    eq_cumsum  = eq_mask.to(torch.int64).cumsum(dim=1)
    eos_cumsum = eos_mask.to(torch.int64).cumsum(dim=1)

    # A token is part of an answer segment if it is:
    #   - After an eq_token has occurred (i.e. eq_cumsum > eos_cumsum)
    #   - Not the eq token itself
    #   - Not an eos token
    #   - Not a padding token
    valid_mask = (eq_cumsum > eos_cumsum) & (~eq_mask) & (~eos_mask) & (~pad_mask)

    # ---------------------------
    # Compute the cross entropy loss over valid tokens.
    # ---------------------------
    batch_size, seq_len, vocab_size = model_output.size()
    # Flatten the tensors so that we can index only the valid positions.
    logits_flat = model_output.view(-1, vocab_size)       # shape: [batch*seq_len, vocab_size]
    labels_flat = labels.view(-1)                           # shape: [batch*seq_len]
    valid_mask_flat = valid_mask.view(-1)                   # shape: [batch*seq_len]

    assert valid_mask_flat.sum() > 0, "No valid tokens found in the batch."
    # Compute the token-level cross entropy loss only for positions in valid_mask.
    loss = F.cross_entropy(logits_flat[valid_mask_flat],
                            labels_flat[valid_mask_flat],
                            reduction='mean')

    # ---------------------------
    # Compute the number of "correct" documents.
    # A document is correct if for all valid tokens in that document, the model's prediction is correct.
    # ---------------------------
    # Get token-level predictions by taking the argmax over the vocab dimension.
    predictions = model_output.argmax(dim=-1)  # shape: [batch, seq_len]
    # Compute a boolean tensor indicating if each token is correctly predicted.
    token_correct = (predictions == labels)  # shape: [batch, seq_len]

    # To group tokens by document, we need a unique group id for each document.
    # We use the fact that within a sequence, valid tokens in a document appear after an eq_token
    # and before the subsequent eos_token. Here we use eos_cumsum as the document identifier.
    # First, create a tensor with the batch indices for each token.
    batch_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand(batch_size, seq_len)

    # Now, restrict to the valid answer tokens.
    valid_batch_idx = batch_idx[valid_mask]       # shape: [N_valid]
    valid_eos_cumsum = eos_cumsum[valid_mask]       # shape: [N_valid]

    # Combine batch index and document id to create a unique group id per document.
    # Use an offset (here, seq_len is safe) so that different batches don't conflict.
    offset = seq_len
    group_ids = valid_batch_idx * offset + valid_eos_cumsum  # shape: [N_valid]

    # For each valid token, get whether it was predicted correctly.
    valid_correct = token_correct[valid_mask].to(torch.int64)  # convert booleans to 0/1

    # Now, we want to group these tokens by group_ids and check per group if all tokens are correct.
    # We use torch.unique to get unique document ids and group information.
    unique_groups, group_inverse, group_counts = torch.unique(group_ids, return_inverse=True, return_counts=True)
    # Initialize a tensor to accumulate correct counts for each document.
    group_correct_sum = torch.zeros_like(group_counts, device=device, dtype=torch.int64)
    # Scatter-add the correctness values into their respective groups.
    group_correct_sum = group_correct_sum.scatter_add(0, group_inverse, valid_correct)

    # A document is correct if the number of correct tokens equals the total token count in that document.
    # (Note: Documents with zero answer tokens do not appear in unique_groups and hence are not counted.)
    correct_docs = (group_correct_sum == group_counts)
    num_correct_docs = utils.safe_int_item(correct_docs.sum())  # Total count of correct documents

    return loss, num_correct_docs

def get_loss_factory(
             eq_token_id: int,
             pad_token_id: int,
             eos_token_id: int,)->Callable[[Tensor, Tensor], Tuple[Tensor, int]]:
    global _eq_token_id, _pad_token_id, _eos_token_id

    _eq_token_id, _pad_token_id, _eos_token_id = eq_token_id, pad_token_id, eos_token_id

    return get_loss