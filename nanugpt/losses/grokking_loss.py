import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple

def get_loss(model_output: Tensor,
             labels: Tensor,
             eq_token_id: int,
             pad_token_id: int,
             eos_token_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute cross entropy loss (only over tokens that are between eq and eos in each document)
    and count the number of documents that are entirely correct (i.e. every valid token is predicted correctly).

    Args:
      model_output: Tensor of shape [B, T, vocab_size].
      labels: Tensor of shape [B, T] (shifted by 1 for autoregressive modeling).
      eq_token_id: int, the token id marking the start of answer region.
      eos_token_id: int, the token id marking end of a document.
      pad_token_id: int, the token id used for padding.

    Returns:
      A tuple (ce_loss, num_correct_docs) where:
        - ce_loss is the cross entropy loss over the valid tokens.
        - num_correct_docs is the count (as a scalar tensor) of documents where all valid tokens were predicted correctly.
    """
    B, T, _ = model_output.shape

    # Create masks for the special tokens.
    eq_mask  = labels.eq(eq_token_id)
    eos_mask = labels.eq(eos_token_id)
    pad_mask = labels.eq(pad_token_id)

    # Compute cumulative counts along the time dimension.
    # For each token, eq_cumsum counts how many eq tokens have been seen so far,
    # and eos_cumsum counts how many eos tokens have been seen.
    eq_cumsum  = torch.cumsum(eq_mask.to(torch.long), dim=1)
    eos_cumsum = torch.cumsum(eos_mask.to(torch.long), dim=1)

    # For a well-formed document, tokens before the eq token have eq_cumsum == eos_cumsum.
    # After the eq token (and before the eos token) we have eq_cumsum > eos_cumsum.
    # We then exclude the eq and eos tokens themselves and any pad tokens.
    valid_mask = (eq_cumsum > eos_cumsum) & (~eq_mask) & (~eos_mask) & (~pad_mask)

    # Compute cross-entropy loss only on valid tokens.
    # model_output[valid_mask] returns a tensor of shape [N, vocab_size],
    # and labels[valid_mask] returns the corresponding target tokens.
    if valid_mask.sum() > 0:
        ce_loss = F.cross_entropy(model_output[valid_mask], labels[valid_mask])
    else:
        ce_loss = torch.tensor(0.0, device=model_output.device)

    # ---- Compute document-level correctness ----
    #
    # We now want to check, for each document, whether *all* valid tokens were predicted correctly.
    # We first need a “document id” per token. We define a document by the tokens that lie
    # between eos tokens (ignoring any pad tokens). In other words, we use a cumulative sum on eos tokens.
    eos_nonpad_mask = eos_mask & (~pad_mask)
    # This gives a doc id per token. (The first document will have doc id 0.)
    doc_ids = torch.cumsum(eos_nonpad_mask.to(torch.long), dim=1)

    # Get the model's predicted token for each position.
    preds = model_output.argmax(dim=-1)

    # Create a mask for tokens that are both valid (i.e. in the answer region)
    # and predicted correctly.
    correct_mask = (preds == labels) & valid_mask

    # To group tokens by document *across the whole batch*, we create a global document id.
    # For each batch element, we assume there are at most T+1 documents.
    batch_idx = torch.arange(B, device=labels.device).unsqueeze(1).expand(B, T)
    global_doc_id = batch_idx * (T + 1) + doc_ids  # shape: [B, T]

    # Flatten our tensors so that we can aggregate per document.
    flat_doc_ids  = global_doc_id.reshape(-1)            # [B*T]
    flat_valid    = valid_mask.reshape(-1).to(torch.long)  # [B*T]
    flat_correct  = correct_mask.reshape(-1).to(torch.long)  # [B*T]

    num_docs = B * (T + 1)
    # For each global document, count the number of valid tokens and
    # the number of tokens predicted correctly.
    doc_valid_counts = torch.zeros(num_docs, device=labels.device, dtype=torch.long)
    doc_correct_counts = torch.zeros(num_docs, device=labels.device, dtype=torch.long)

    # scatter_add accumulates values from flat_valid into the appropriate document slots.
    doc_valid_counts = doc_valid_counts.scatter_add(0, flat_doc_ids, flat_valid)
    doc_correct_counts = doc_correct_counts.scatter_add(0, flat_doc_ids, flat_correct)

    # A document is considered correct if it has at least one valid token and
    # the number of correct tokens equals the total number of valid tokens.
    doc_correct_indicator = (doc_valid_counts > 0) & (doc_valid_counts == doc_correct_counts)
    num_correct_docs = doc_correct_indicator.sum()

    return ce_loss, num_correct_docs


