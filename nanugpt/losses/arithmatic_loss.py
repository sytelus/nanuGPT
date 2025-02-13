import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple

def get_loss(model_output: Tensor, labels: Tensor, eq_token_id: int, pad_token_id: int, eos_token_id: int) -> Tuple[Tensor, Tensor]:
    """
    Computes (1) cross-entropy loss and (2) the number of correct documents.

    model_output: Tensor of shape [B, L, V] (logits over vocab)
    labels:       Tensor of shape [B, L] (token ids, shifted by one for autoregression)

    For each document (a segment in a sequence ending with eos_token_id, containing exactly one eq_token_id),
    we only compute loss on tokens between eq_token_id and eos_token_id (both excluded).

    A document is considered “correct” if the predicted tokens (via argmax) exactly match the label tokens
    in the answer region.

    Sequences may have padding at the end (pad_token_id) which do not belong to any document.
    """
    B, L, V = model_output.shape
    device = labels.device  # ensure all new tensors are on the same device

    # ---------------------------------------------------------------
    # Create a mask for tokens that belong to an answer region.
    # We do this by computing, for each position, the last seen eq token and eos token.
    #
    # For each sequence, define:
    #   last_eq[i]  = maximum index j <= i such that labels[j] == eq_token_id (or -1 if none)
    #   last_eos[i] = maximum index j <= i such that labels[j] == eos_token_id (or -1 if none)
    #
    # Then, a token at position i is in the answer region if:
    #   (a) It is not pad, eq, or eos,
    #   (b) And it comes after an eq token in the current document,
    #       i.e. last_eq[i] > last_eos[i]   (since eos resets the document)
    # ---------------------------------------------------------------
    # positions: tensor [B, L] with values 0,1,...,L-1
    positions = torch.arange(L, device=device).unsqueeze(0).expand(B, L)

    # For eq tokens: record position if token equals eq_token_id, else -1.
    eq_positions = torch.where(labels == eq_token_id, positions, torch.full_like(positions, -1))
    # Cumulative max gives, at each position, the last (largest) eq token position seen so far.
    last_eq = eq_positions.cummax(dim=1).values

    # Similarly for eos tokens.
    eos_positions = torch.where(labels == eos_token_id, positions, torch.full_like(positions, -1))
    last_eos = eos_positions.cummax(dim=1).values

    # A token is valid for loss if:
    #  (1) it is not a pad token,
    #  (2) it is not the eq or eos token (we exclude these from loss),
    #  (3) and it is in the region after an eq in the current document: last_eq > last_eos.
    valid_loss_mask = (labels != pad_token_id) & (labels != eq_token_id) & (labels != eos_token_id) & (last_eq > last_eos)

    # ---------------------------------------------------------------
    # Compute cross-entropy loss only on the valid tokens.
    # We index model_output and labels with the mask.
    # If no token qualifies (mask is empty) we return a loss of 0.
    # ---------------------------------------------------------------
    if valid_loss_mask.sum() > 0:
        # model_output has shape [B, L, V] and valid_loss_mask is [B, L] --> result is [N, V]
        logits = model_output[valid_loss_mask]
        target = labels[valid_loss_mask]
        loss = F.cross_entropy(logits, target, reduction='mean')
    else:
        loss = torch.tensor(0.0, device=device)

    # ---------------------------------------------------------------
    # Compute number of correct documents.
    # For each sequence in the batch, we identify each document using the location of its eq and eos tokens.
    # For each eq token, we take the first eos token that follows it.
    # We then compare the predicted tokens (argmax from model_output) with the label tokens for the answer region.
    # A document is correct only if all tokens in the answer region match.
    # (If there is no answer token (i.e. empty region), we count it as correct.)
    # ---------------------------------------------------------------
    preds = torch.argmax(model_output, dim=-1)  # shape [B, L]
    correct_docs = 0
    for b in range(B):
        # Find positions of eq and eos tokens in this sequence.
        eq_indices = (labels[b] == eq_token_id).nonzero(as_tuple=False).flatten()
        eos_indices = (labels[b] == eos_token_id).nonzero(as_tuple=False).flatten()

        # If there are no eq tokens or eos tokens, skip this sequence.
        if eq_indices.numel() == 0 or eos_indices.numel() == 0:
            continue

        # For each eq token (document start), find the first eos token that comes after it.
        # (Assumes documents occur in order and each document has exactly one eq and one eos.)
        for eq_idx in eq_indices.tolist():
            # Find the first eos index greater than eq_idx.
            eos_after = eos_indices[eos_indices > eq_idx]
            if eos_after.numel() == 0:
                continue  # no eos token after this eq token; skip
            eos_idx = int(eos_after[0].item())

            # The answer region is the tokens between eq and eos (excluding both).
            start = eq_idx + 1
            end = eos_idx  # end index is not included

            # If the answer region is empty, we count it as correct.
            if end - start <= 0:
                correct_docs += 1
            else:
                pred_segment = preds[b, start:end]
                label_segment = labels[b, start:end]
                if torch.equal(pred_segment, label_segment):
                    correct_docs += 1

    # Return correct_docs as a tensor on the same device.
    correct_docs_tensor = torch.tensor(correct_docs, device=device, dtype=torch.long)

    return loss, correct_docs_tensor
