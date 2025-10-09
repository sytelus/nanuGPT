#!/usr/bin/env python3
"""Benchmark TeLlama3Model configurations with timing and memory stats."""

from __future__ import annotations

import argparse
import math
import statistics
import time
from dataclasses import asdict
from typing import Dict, Iterable, List, Optional, Tuple
import sys

import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from nanugpt.models import te_llama3
from nanugpt.models.te_llama3 import LlamaConfig, TeLlama3Model


console = Console()


def _active_debugger_name() -> Optional[str]:
    """Return name of active debugger trace if any."""
    gettrace = getattr(sys, "gettrace", None)
    if gettrace is None:
        return None
    trace = gettrace()
    if trace is None:
        return None
    module = getattr(trace, "__module__", "")
    qualname = getattr(trace, "__qualname__", getattr(trace, "__name__", ""))
    identifier = ".".join(filter(None, [module, qualname])).strip(".")
    return identifier or repr(trace)


def _torch_compile_model(
    model: TeLlama3Model,
    device: torch.device,
) -> TeLlama3Model:
    debugger = _active_debugger_name()
    if debugger:
        console.print(
            f"[bold yellow]torch.compile is disabled while debugger '{debugger}' is active. "
            "Rerun without the debugger or pass --no-compile.[/bold yellow]"
        )
        raise RuntimeError(
            f"torch.compile is not supported while a debugger trace ({debugger}) is active. "
            "Rerun without the debugger or pass --no-compile."
        )
    if device.type != "cuda":
        raise RuntimeError(
            "torch.compile() with Transformer Engine layers currently requires a CUDA device."
        )

    dynamo_module = None
    original_cudagraphs: Optional[bool] = None
    try:
        import torch._dynamo as dynamo_module  # type: ignore[attr-defined]
    except Exception:
        dynamo_module = None

    if dynamo_module and hasattr(dynamo_module.config, "use_cudagraphs"):
        original_cudagraphs = dynamo_module.config.use_cudagraphs
        dynamo_module.config.use_cudagraphs = False

    try:
        return torch.compile(model, mode="reduce-overhead", dynamic=True)
    except TypeError as exc:
        if "unhashable type: 'dict'" in str(exc):
            raise RuntimeError(
                "torch.compile triggered PyDevd 'unhashable dict' failures. "
                "Run the benchmark outside the debugger or use --no-compile."
            ) from exc
        raise
    finally:
        if dynamo_module and original_cudagraphs is not None:
            dynamo_module.config.use_cudagraphs = original_cudagraphs


def discover_configs(prefix: str = "TE_") -> Dict[str, LlamaConfig]:
    """Discover LlamaConfig instances in te_llama3 that match the prefix."""
    configs: Dict[str, LlamaConfig] = {}
    for name in dir(te_llama3):
        if not name.startswith(prefix):
            continue
        value = getattr(te_llama3, name)
        if isinstance(value, LlamaConfig):
            configs[name] = value
    return configs


def format_bytes_as_mb(num_bytes: float) -> float:
    """Convert bytes to megabytes."""
    return num_bytes / (1024 ** 2)


def average_or_zero(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.mean(values) if values else 0.0


def aggregate_metrics(records: List[Dict[str, float]]) -> Dict[str, float]:
    """Aggregate timing and memory stats across iterations."""
    summary = {
        "forward_time": average_or_zero(r["forward_time"] for r in records),
        "backward_time": average_or_zero(r["backward_time"] for r in records),
        "forward_memory": average_or_zero(r["forward_memory"] for r in records),
        "backward_memory": average_or_zero(r["backward_memory"] for r in records),
    }
    return summary


def run_single_iteration(
    model: TeLlama3Model,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
) -> Tuple[float, float, float, float]:
    """Run one forward/backward pass returning timing (s) and memory (bytes)."""
    memory_forward = float("nan")
    memory_backward = float("nan")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        mem_before_forward = torch.cuda.memory_allocated(device)
    else:
        mem_before_forward = 0

    torch.cuda.synchronize(device) if device.type == "cuda" else None
    start_forward = time.perf_counter()
    logits, _, _ = model(input_ids, return_logits=True)
    torch.cuda.synchronize(device) if device.type == "cuda" else None
    forward_time = time.perf_counter() - start_forward

    if device.type == "cuda":
        memory_forward = torch.cuda.max_memory_allocated(device) - mem_before_forward
        torch.cuda.reset_peak_memory_stats(device)
        mem_before_backward = torch.cuda.memory_allocated(device)
    else:
        mem_before_backward = 0

    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)), labels.view(-1)
    )

    torch.cuda.synchronize(device) if device.type == "cuda" else None
    start_backward = time.perf_counter()
    loss.backward()
    torch.cuda.synchronize(device) if device.type == "cuda" else None
    backward_time = time.perf_counter() - start_backward

    if device.type == "cuda":
        memory_backward = torch.cuda.max_memory_allocated(device) - mem_before_backward

    model.zero_grad(set_to_none=True)

    return forward_time, backward_time, memory_forward, memory_backward


def benchmark_config(
    name: str,
    config: LlamaConfig,
    device: torch.device,
    seq_len: int,
    batch_size: int,
    warmup: int,
    steps: int,
    do_compile: bool,
) -> Tuple[Dict[str, float], int]:
    """Benchmark a configuration, optionally using torch.compile."""
    torch.manual_seed(42)

    model = TeLlama3Model(config, get_loss=None)
    param_count = sum(p.numel() for p in model.parameters())
    model = model.to(device)

    if do_compile:
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile is not available in this PyTorch build.")
        model = _torch_compile_model(model, device)

    vocab_size = config.vocab_size
    context_length = min(seq_len, config.block_size)

    input_ids = torch.randint(
        low=0, high=vocab_size, size=(batch_size, context_length), device=device
    )
    labels = torch.randint(
        low=0, high=vocab_size, size=(batch_size, context_length), device=device
    )

    records: List[Dict[str, float]] = []
    total_iters = warmup + steps

    for iteration in range(total_iters):
        try:
            forward_time, backward_time, mem_forward, mem_backward = run_single_iteration(
                model, input_ids, labels, device
            )
        except torch.cuda.OutOfMemoryError:
            console.print(
                f"[bold red]OOM encountered for {name} (compile={do_compile}). "
                f"Consider reducing --batch-size or --seq-length.[/bold red]"
            )
            raise

        if iteration >= warmup:
            records.append(
                {
                    "forward_time": forward_time,
                    "backward_time": backward_time,
                    "forward_memory": mem_forward,
                    "backward_memory": mem_backward,
                }
            )

    return aggregate_metrics(records), param_count


def make_config_panel(name: str, config: LlamaConfig, param_count: int) -> Panel:
    config_table = Table(show_header=False, expand=True)
    for key, value in asdict(config).items():
        config_table.add_row(f"[bold]{key}[/bold]", str(value))

    config_table.add_row("[bold]parameters[/bold]", f"{param_count:,} ({param_count/1e6:.2f}M)")
    return Panel(config_table, title=name, subtitle="core configuration", expand=False)


def print_results(
    name: str,
    config: LlamaConfig,
    eager_metrics: Dict[str, float],
    compiled_metrics: Dict[str, float] | None,
    device: torch.device,
    param_count: int,
):
    console.print()
    console.print(make_config_panel(name, config, param_count))

    table = Table(title=f"{name} performance", expand=True)
    table.add_column("Mode", justify="center", style="cyan")
    table.add_column("Avg Fwd (ms)", justify="right")
    table.add_column("Avg Bwd (ms)", justify="right")
    table.add_column("Fwd Mem (MB)", justify="right")
    table.add_column("Bwd Mem (MB)", justify="right")

    def add_row(mode_name: str, metrics: Dict[str, float]):
        forward_ms = metrics["forward_time"] * 1e3
        backward_ms = metrics["backward_time"] * 1e3
        if device.type == "cuda":
            fwd_mem = format_bytes_as_mb(metrics["forward_memory"])
            bwd_mem = format_bytes_as_mb(metrics["backward_memory"])
        else:
            fwd_mem = bwd_mem = float("nan")
        fwd_display = "n/a" if math.isnan(fwd_mem) else f"{fwd_mem:8.2f}"
        bwd_display = "n/a" if math.isnan(bwd_mem) else f"{bwd_mem:8.2f}"
        table.add_row(
            mode_name,
            f"{forward_ms:8.2f}",
            f"{backward_ms:8.2f}",
            fwd_display,
            bwd_display,
        )

    add_row("eager", eager_metrics)
    if compiled_metrics is not None:
        add_row("compiled", compiled_metrics)

    console.print(table)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark TeLlama3Model configs with timing and memory stats."
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for synthetic data.")
    parser.add_argument(
        "--seq-length",
        type=int,
        default=512,
        help="Sequence length for synthetic data (capped at config block_size).",
    )
    parser.add_argument("--warmup", type=int, default=1, help="Number of warmup iterations.")
    parser.add_argument("--steps", type=int, default=5, help="Number of timed iterations.")
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Skip torch.compile measurements.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on (defaults to cuda if available else cpu).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Run a single config (name must match TE_* attribute).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type != "cuda":
        console.print(
            "[bold yellow]CUDA device not detected. "
            "Transformer Engine layers generally require GPU support.[/bold yellow]"
        )

    configs = discover_configs()

    if args.config:
        if args.config not in configs:
            raise ValueError(f"Unknown config '{args.config}'. Available: {', '.join(sorted(configs))}")
        configs = {args.config: configs[args.config]}

    if not configs:
        console.print("[bold red]No TE_ prefixed LlamaConfig found in te_llama3.[/bold red]")
        return

    for name, config in configs.items():
        try:
            eager_metrics, param_count = benchmark_config(
                name=name,
                config=config,
                device=device,
                seq_len=args.seq_length,
                batch_size=args.batch_size,
                warmup=args.warmup,
                steps=args.steps,
                do_compile=False,
            )
        except Exception as exc:
            console.print(f"[bold red]Failed to benchmark {name} (eager). Reason: {exc}[/bold red]")
            continue

        compiled_metrics = None
        if not args.no_compile:
            try:
                compiled_metrics, _ = benchmark_config(
                    name=name,
                    config=config,
                    device=device,
                    seq_len=args.seq_length,
                    batch_size=args.batch_size,
                    warmup=max(args.warmup, 1),
                    steps=args.steps,
                    do_compile=True,
                )
            except Exception as exc:
                console.print(f"[bold yellow]Skipping compile benchmark for {name}: {exc}[/bold yellow]")
                compiled_metrics = None

        print_results(
            name=name,
            config=config,
            eager_metrics=eager_metrics,
            compiled_metrics=compiled_metrics,
            device=device,
            param_count=param_count,
        )

        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
