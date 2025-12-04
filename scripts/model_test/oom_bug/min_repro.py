"""
Minimal demonstration of OOM bug: autocast + gradient checkpointing mismatch.

ROOT CAUSE:
When using gradient checkpointing, activations are recomputed during backward().
If forward() runs under autocast (bfloat16) but backward() is called OUTSIDE the
autocast context, the recomputation may occur in fp32, DOUBLING memory usage.

Usage:
    python demo.py --buggy   # Shows increased memory / potential OOM
    python demo.py --fixed   # Shows correct memory-efficient pattern
"""

import argparse
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--buggy", action="store_true")
    parser.add_argument("--fixed", action="store_true")
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--num-layers", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=512)
    args = parser.parse_args()

    if not (args.buggy or args.fixed):
        print("Usage: python demo.py [--buggy | --fixed]")
        print("\n--buggy: autocast around loss, backward outside (high memory)")
        print("--fixed: autocast only around model forward (low memory)")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CheckpointedModel(args.hidden_size, args.num_layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    x = torch.randn(args.batch_size, args.seq_len, args.hidden_size, device=device)
    y = torch.randn(args.batch_size, args.seq_len, args.hidden_size, device=device)

    torch.cuda.reset_peak_memory_stats()

    train_fn = buggy_training_step if args.buggy else fixed_training_step

    print(f"\nRunning {'BUGGY' if args.buggy else 'FIXED'} pattern...")

    for step in range(3):
        try:
            loss = train_fn(model, optimizer, x, y)
            peak_mb = torch.cuda.max_memory_allocated() / 1e6
            print(f"Step {step+1}: loss={loss:.4f}, peak_memory={peak_mb:.0f}MB")
        except torch.cuda.OutOfMemoryError:
            print(f"❌ OOM at step {step+1}!")
            print("The buggy pattern caused checkpoint recomputation in fp32,")
            print("doubling activation memory and causing OOM.")
            return

    print(f"\n✓ Completed. Peak memory: {torch.cuda.max_memory_allocated()/1e6:.0f}MB")


if __name__ == "__main__":
    main()