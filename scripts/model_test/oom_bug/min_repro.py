"""Quick comparison of buggy vs fixed autocast + checkpointing memory use."""
import torch
import torch.nn as nn


class GPTModel(nn.Module):
    """Model using gradient checkpointing to save memory."""

    def __init__(self, hidden_size=1536, num_layers=28, vocab_size=151_936):
        super().__init__()
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

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for layer in self.layers:
            # Gradient checkpointing: recompute activations during backward
            x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
        return self.head(x)


def buggy_training_step(model, optimizer, input_ids, labels):
    """
    BUGGY: Wraps loss computation in autocast, but backward() is OUTSIDE.

    This causes gradient checkpointing to potentially recompute in fp32!
    """
    # Autocast wraps forward, but backward will be outside
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(input_ids)
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
        )

    optimizer.zero_grad()
    loss.backward()  # Outside autocast - checkpoint recomputes may use fp32!
    optimizer.step()
    return loss.item()


def fixed_training_step(model, optimizer, input_ids, labels):
    """
    FIXED: Autocast only around model forward, OR include backward inside autocast.
    """
    # Autocast only around model forward
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(input_ids)

    # Loss computation can be in fp32 - that's fine
    loss = nn.functional.cross_entropy(
        logits.float().view(-1, logits.size(-1)),
        labels.view(-1),
    )

    optimizer.zero_grad()
    loss.backward()  # Checkpoint state was captured correctly
    optimizer.step()
    return loss.item()


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for memory comparison demo.")

    device = torch.device("cuda")
    cfg = dict(
        hidden_size=1536,  # Qwen 2.5 1.5B
        num_layers=28,     # Qwen 2.5 1.5B
        vocab_size=151_936,
        batch_size=4,
        seq_len=512,
    )

    def run(label, train_fn):
        model = GPTModel(cfg["hidden_size"], cfg["num_layers"], cfg["vocab_size"]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        input_ids = torch.randint(
            cfg["vocab_size"],
            (cfg["batch_size"], cfg["seq_len"]),
            device=device,
        )
        labels = torch.randint_like(input_ids, cfg["vocab_size"])

        torch.cuda.reset_peak_memory_stats()
        for _ in range(3):
            train_fn(model, optimizer, input_ids, labels)
        peak_mb = torch.cuda.max_memory_allocated() / 1e6
        torch.cuda.empty_cache()
        print(f"{label} peak memory: {peak_mb:.0f} MB")

    run("BUGGY ", buggy_training_step)
    run("FIXED ", fixed_training_step)


if __name__ == "__main__":
    main()
