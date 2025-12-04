"""Quick comparison of buggy vs fixed autocast + checkpointing memory use."""
import torch
import torch.nn as nn


class CheckpointedModel(nn.Module):
    """Model using gradient checkpointing to save memory."""

    def __init__(self, hidden_size=2048, num_layers=16):
        super().__init__()
        self.embed = nn.Linear(hidden_size, hidden_size)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4),
                nn.GELU(),
                nn.Linear(hidden_size * 4, hidden_size),
                nn.LayerNorm(hidden_size),
            )
            for _ in range(num_layers)
        ])
        self.head = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            # Gradient checkpointing: recompute activations during backward
            x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
        return self.head(x)


def buggy_training_step(model, optimizer, x, y):
    """
    BUGGY: Wraps loss computation in autocast, but backward() is OUTSIDE.

    This causes gradient checkpointing to potentially recompute in fp32!
    """
    # Autocast wraps forward, but backward will be outside
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(x)
        loss = nn.functional.mse_loss(output, y)

    optimizer.zero_grad()
    loss.backward()  # Outside autocast - checkpoint recomputes may use fp32!
    optimizer.step()
    return loss.item()


def fixed_training_step(model, optimizer, x, y):
    """
    FIXED: Autocast only around model forward, OR include backward inside autocast.
    """
    # Autocast only around model forward
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(x)

    # Loss computation can be in fp32 - that's fine
    loss = nn.functional.mse_loss(output.float(), y)

    optimizer.zero_grad()
    loss.backward()  # Checkpoint state was captured correctly
    optimizer.step()
    return loss.item()


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for memory comparison demo.")

    device = torch.device("cuda")
    cfg = dict(hidden_size=2048, num_layers=16, batch_size=32, seq_len=512)

    def run(label, train_fn):
        model = CheckpointedModel(cfg["hidden_size"], cfg["num_layers"]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        x = torch.randn(cfg["batch_size"], cfg["seq_len"], cfg["hidden_size"], device=device)
        y = torch.randn_like(x)

        torch.cuda.reset_peak_memory_stats()
        for _ in range(3):
            train_fn(model, optimizer, x, y)
        peak_mb = torch.cuda.max_memory_allocated() / 1e6
        torch.cuda.empty_cache()
        print(f"{label} peak memory: {peak_mb:.0f} MB")

    run("BUGGY ", buggy_training_step)
    run("FIXED ", fixed_training_step)


if __name__ == "__main__":
    main()
