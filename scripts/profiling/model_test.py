"""
# For memory visulization, run:
# python _memory_viz.py trace_plot "memory.pickle" -o memory.html
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

class GPTModel(nn.Module):
    def __init__(
        self,
        hidden_size: int = 1536,
        num_layers: int = 28,
        vocab_size: int = 151_936,
        activation_checkpointing: bool = True,
    ):
        super().__init__()
        self.activation_checkpointing = activation_checkpointing
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4),
                nn.GELU(),
                nn.Linear(hidden_size * 4, hidden_size),
                nn.LayerNorm(hidden_size),
            )
            for _ in range(num_layers)
        ])
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(
        self,
        input_ids: Tensor,
        labels: Optional[Tensor] = None,
        return_logits: bool = True,
    ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        x = self.embed(input_ids)
        for layer in self.layers:
            # Gradient checkpointing: recompute activations during backward
            if self.activation_checkpointing:
                x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        logits = self.head(x)

        if return_logits:
            loss: Optional[Tensor] = None
            if labels is not None:
                loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            return logits, loss

        loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        return None, loss

def training_step(
    model: GPTModel,
    optimizer: torch.optim.Optimizer,
    input_ids: Tensor,
    labels: Tensor,
) -> float:
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        _, loss = model(input_ids, labels, return_logits=False)

    optimizer.zero_grad()
    loss.backward()  # Checkpoint state was captured correctly
    optimizer.step()
    return loss.item()

def main() -> None:
    device = torch.device("cuda")
    cfg: Dict[str, int | bool] = dict(
        hidden_size=1536,
        num_layers=28,
        vocab_size=151_936,
        batch_size=59,
        seq_len=512,
        activation_checkpointing=True,
    )

    model = GPTModel(cfg["hidden_size"], cfg["num_layers"], cfg["vocab_size"], cfg["activation_checkpointing"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    input_ids = torch.randint(cfg["vocab_size"], (cfg["batch_size"], cfg["seq_len"]), device=device)
    labels = torch.randint_like(input_ids, cfg["vocab_size"])

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats()
    torch.cuda.memory._record_memory_history(max_entries=100000)
    for _ in range(3):
        training_step(model, optimizer, input_ids, labels)
    torch.cuda.synchronize()

    torch.cuda.memory._dump_snapshot("memory.pickle")
    torch.cuda.memory._record_memory_history(enabled=None)
    stats = torch.cuda.memory_stats()
    active_mb = stats["active_bytes.all.allocated"] / 1e6  # matches Active Memory Timeline
    reserved_mb = stats["reserved_bytes.all.peak"] / 1e6

    torch.cuda.empty_cache()

    print(f"peak active: {active_mb:.0f} MB | peak reserved: {reserved_mb:.0f} MB")

if __name__ == "__main__":
    main()
