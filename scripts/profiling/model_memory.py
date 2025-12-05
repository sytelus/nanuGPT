import pprint
import tempfile
from typing import Dict, Optional, Tuple

import os
import torch
import torch.accelerator as accel
import torch.nn as nn
from torch import Tensor

from nanugpt.acc_profile import AccProfile

class MLP(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.c_fc = nn.Linear(hidden_size, hidden_size * 4)
        self.act = nn.GELU()
        self.c_proj = nn.Linear(hidden_size * 4, hidden_size)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.drop(self.c_proj(self.act(self.c_fc(x))))

class GPTBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5

        self.ln_1 = nn.LayerNorm(hidden_size)
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(dropout)

        self.ln_2 = nn.LayerNorm(hidden_size)
        self.mlp = MLP(hidden_size, dropout)

    def forward(self, x: Tensor) -> Tensor:
        h = self.ln_1(x)
        q = self.q_proj(h).view(h.size(0), h.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(h.size(0), h.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(h.size(0), h.size(1), self.num_heads, self.head_dim).transpose(1, 2)

        attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.attn_dropout.p if self.training else 0.0, is_causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(h.size(0), h.size(1), -1)
        x = x + self.out_proj(attn_out)
        return x + self.mlp(self.ln_2(x))

class GPTModel(nn.Module):
    def __init__(
        self,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        vocab_size: int = 50_257,
        activation_checkpointing: bool = True,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        device: torch.device | str | None = None,
    ):
        super().__init__()
        self.activation_checkpointing = activation_checkpointing
        self.max_seq_len = max_seq_len
        weight = torch.empty((vocab_size, hidden_size))
        self.embed = nn.Embedding.from_pretrained(weight, freeze=False)
        pos_weight = torch.empty((max_seq_len, hidden_size))
        self.pos_embed = nn.Embedding.from_pretrained(pos_weight, freeze=False)
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList([GPTBlock(hidden_size, num_heads, dropout) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids: Tensor, labels: Optional[Tensor] = None, return_logits: bool = True) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        bsz, seq_len = input_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}")

        pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, seq_len)
        x = self.drop(self.embed(input_ids) + self.pos_embed(pos))

        for layer in self.layers:
            fn = lambda y: layer(y)
            x = torch.utils.checkpoint.checkpoint(fn, x, use_reentrant=False) if self.activation_checkpointing else layer(x)  # type: ignore[arg-type]

        x = self.ln_f(x)
        logits = self.head(x)

        loss: Optional[Tensor] = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

        return (logits, loss) if return_logits else (None, loss)


def training_step(model: GPTModel, optimizer: torch.optim.Optimizer, input_ids: Tensor, labels: Tensor) -> float:
    device_type = input_ids.device.type
    autocast_ctx = torch.autocast(device_type=device_type, dtype=torch.bfloat16)
    with autocast_ctx:
        _, loss = model(input_ids, labels, return_logits=False)

    optimizer.zero_grad()
    loss.backward()  # Checkpoint state was captured correctly
    optimizer.step()
    return loss.item()


def main() -> None:
    device = torch.device(accel.current_accelerator(check_available=True) or "cpu")
    torch.set_default_device(device)
    cfg: Dict[str, int | bool] = dict(hidden_size=1536,
                                      num_layers=36,
                                      num_heads=12,
                                      vocab_size=151_643,
                                      batch_size=32,
                                      seq_len=512,
                                      activation_checkpointing=True)

    model = GPTModel(
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        vocab_size=cfg["vocab_size"],
        activation_checkpointing=cfg["activation_checkpointing"], # type: ignore[arg-type]
        device=device,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    input_ids = torch.randint(cfg["vocab_size"], (cfg["batch_size"], cfg["seq_len"]), device=device)
    labels = torch.randint_like(input_ids, cfg["vocab_size"])

    model.compile()
    model.train()

    # Warm-up
    for _ in range(3):
        training_step(model, optimizer, input_ids, labels)

    with AccProfile() as prof:
        for _ in range(3):
            training_step(model, optimizer, input_ids, labels)

    pprint.pprint(prof.memory_stats)

    out_dir = os.path.join(os.environ.get("OUT_DIR", tempfile.gettempdir()), "model_memory")
    os.makedirs(out_dir, exist_ok=True)
    path = prof.save_html(os.path.join(out_dir, "model_memory_profile.html"))
    print(f"Memory snapshot saved to `{path}`")

if __name__ == "__main__":
    main()
