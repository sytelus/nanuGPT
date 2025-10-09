#!/usr/bin/env python3
"""Benchmark TeLlama3Model configurations with timing and memory stats."""

from __future__ import annotations

import argparse
import math
import statistics
import time
import warnings
from functools import lru_cache
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch
from rich.console import Console
from rich.table import Table

from nanugpt.models import te_llama3
from nanugpt.models.te_llama3 import LlamaConfig, TeLlama3Model


console = Console()

SUPPRESS_COMPILE_WARNINGS = True  # Set to False to see full torch.compile warning output.


@dataclass
class BenchmarkSummary:
    name: str
    config: LlamaConfig
    param_stats: Dict[str, int]
    eager_metrics: Optional[Dict[str, float]]
    compiled_metrics: Optional[Dict[str, float]]
    eager_oom: bool = False
    compiled_oom: bool = False


def _torch_compile_model(
    model: TeLlama3Model,
    device: torch.device,
) -> TeLlama3Model:
    if device.type != "cuda":
        raise RuntimeError(
            "torch.compile() with Transformer Engine layers currently requires a CUDA device."
        )

    if SUPPRESS_COMPILE_WARNINGS:
        _configure_compile_environment()
    return torch.compile(model, mode="max-autotune", dynamic=True)

@lru_cache(maxsize=1)
def _configure_compile_environment() -> None:
    import logging
    warning_patterns = [
        r"Dynamo detected a call to a `functools\.lru_cache`-wrapped function",
        r"Dynamo does not know how to trace the builtin `<unknown module>\.ArgsKwargs\.__new__`",
        r"Dynamo does not know how to trace the builtin `transformer_engine_torch\.PyCapsule\.fused_rope_forward`",
        r"The CUDA Graph is empty\.",
    ]
    for pattern in warning_patterns:
        warnings.filterwarnings("ignore", message=pattern, category=UserWarning)

    logging.getLogger("torch.fx.experimental.symbolic_shapes").setLevel(logging.ERROR)

    console.print(
        "[bold yellow]Suppressing known torch.compile warnings. Use --no-compile to skip compilation entirely.[/bold yellow]"
    )

def discover_configs(prefix: str = "TE_") -> Dict[str, LlamaConfig]:
    """Discover LlamaConfig instances in te_llama3 that match the prefix."""
    return {
        name: value
        for name in dir(te_llama3)
        if name.startswith(prefix) and isinstance(value := getattr(te_llama3, name), LlamaConfig)
    }


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

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start_forward = time.perf_counter()
    logits, _, _ = model(input_ids, return_logits=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
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

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start_backward = time.perf_counter()
    loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
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
) -> Tuple[Optional[Dict[str, float]], Dict[str, int], bool]:
    """Benchmark a configuration, optionally using torch.compile."""
    torch.manual_seed(42)

    model = TeLlama3Model(config, get_loss=None)
    param_total = sum(p.numel() for p in model.parameters())
    param_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    embedding_params = model.tok_embeddings.weight.numel()
    param_stats = {
        "total": param_total,
        "trainable": param_trainable,
        "embedding": embedding_params,
    }
    try:
        model = model.to(device)
    except torch.cuda.OutOfMemoryError:
        console.print(f"[bold red]{name}: OOM while moving model to {device}.[/bold red]")
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return None, param_stats, True

    if do_compile:
        console.print(f"[dim]{name}: compiling model with torch.compile...[/dim]")
        try:
            model = _torch_compile_model(model, device)
        except torch.cuda.OutOfMemoryError:
            console.print(f"[bold red]{name}: OOM while compiling model.[/bold red]")
            if device.type == "cuda":
                torch.cuda.empty_cache()
            return None, param_stats, True

    vocab_size = config.vocab_size
    context_length = min(seq_len, config.block_size)

    try:
        input_ids = torch.randint(
            low=0, high=vocab_size, size=(batch_size, context_length), device=device
        )
        labels = torch.randint(
            low=0, high=vocab_size, size=(batch_size, context_length), device=device
        )
    except torch.cuda.OutOfMemoryError:
        console.print(f"[bold red]{name}: OOM while allocating synthetic data.[/bold red]")
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return None, param_stats, True

    oom = False
    for i in range(max(warmup, 0)):
        console.print(f"[dim]{name}: warmup {i + 1}/{warmup}[/dim]")
        try:
            run_single_iteration(model, input_ids, labels, device)
        except torch.cuda.OutOfMemoryError:
            console.print(f"[bold red]{name}: OOM during warmup (compile={do_compile}).[/bold red]")
            oom = True
            break

    records: List[Dict[str, float]] = []
    if not oom:
        for i in range(max(steps, 0)):
            console.print(f"[dim]{name}: measuring {i + 1}/{steps}[/dim]")
            try:
                forward_time, backward_time, mem_forward, mem_backward = run_single_iteration(
                    model, input_ids, labels, device
                )
            except torch.cuda.OutOfMemoryError:
                console.print(f"[bold red]{name}: OOM during measurement (compile={do_compile}).[/bold red]")
                oom = True
                break
            records.append(
                {
                    "forward_time": forward_time,
                    "backward_time": backward_time,
                    "forward_memory": mem_forward,
                    "backward_memory": mem_backward,
                }
            )

    if oom:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return None, param_stats, True

    return aggregate_metrics(records), param_stats, False


def print_summary_table(results: List[BenchmarkSummary], device: torch.device) -> None:
    if not results:
        console.print("[bold red]No benchmark results to display.[/bold red]")
        return

    table = Table(title="TeLlama3 Benchmark Summary", expand=True)
    table.add_column("Metric", justify="left", style="bold")
    for summary in results:
        table.add_column(summary.name, justify="right")

    def add_row(label: str, extractor: Callable[[BenchmarkSummary], str]) -> None:
        table.add_row(label, *[extractor(summary) for summary in results])

    def fmt_config(attr: str) -> Callable[[BenchmarkSummary], str]:
        return lambda summary: str(getattr(summary.config, attr))

    def fmt_params(key: str) -> Callable[[BenchmarkSummary], str]:
        return lambda summary: f"{summary.param_stats[key] / 1e6:.2f}M"

    def fmt_time(key: str, metrics_type: str) -> Callable[[BenchmarkSummary], str]:
        def _formatter(summary: BenchmarkSummary) -> str:
            metrics = summary.eager_metrics if metrics_type == "eager" else summary.compiled_metrics
            oom = summary.eager_oom if metrics_type == "eager" else summary.compiled_oom
            if oom:
                return "OOM"
            if not metrics or key not in metrics:
                return "n/a"
            value = metrics[key]
            if value is None or math.isnan(value):
                return "n/a"
            return f"{value * 1e3:,.2f}"

        return _formatter

    add_row("n_embd", fmt_config("n_embd"))
    add_row("n_layer", fmt_config("n_layer"))
    add_row("n_head", fmt_config("n_head"))
    add_row("n_kv_heads", fmt_config("n_kv_heads"))
    add_row("block_size", fmt_config("block_size"))
    add_row("vocab_size", fmt_config("vocab_size"))

    add_row("Embedding Params (M)", fmt_params("embedding"))
    add_row("Trainable Params (M)", fmt_params("trainable"))
    add_row("Total Params (M)", fmt_params("total"))

    add_row("Forward ms (eager)", fmt_time("forward_time", "eager"))
    add_row("Backward ms (eager)", fmt_time("backward_time", "eager"))

    add_compile_rows = any(
        summary.compiled_metrics is not None or summary.compiled_oom for summary in results
    )
    if add_compile_rows:
        add_row("Forward ms (compiled)", fmt_time("forward_time", "compiled"))
        add_row("Backward ms (compiled)", fmt_time("backward_time", "compiled"))

    if device.type == "cuda":
        def fmt_mem(key: str, metrics_type: str) -> Callable[[BenchmarkSummary], str]:
            def _formatter(summary: BenchmarkSummary) -> str:
                metrics = summary.eager_metrics if metrics_type == "eager" else summary.compiled_metrics
                oom = summary.eager_oom if metrics_type == "eager" else summary.compiled_oom
                if oom:
                    return "OOM"
                if not metrics or key not in metrics:
                    return "n/a"
                value = metrics[key]
                if value is None or math.isnan(value):
                    return "n/a"
                return f"{format_bytes_as_mb(value):,.2f}"

            return _formatter

        add_row("Forward MB (eager)", fmt_mem("forward_memory", "eager"))
        add_row("Backward MB (eager)", fmt_mem("backward_memory", "eager"))
        if add_compile_rows:
            add_row("Forward MB (compiled)", fmt_mem("forward_memory", "compiled"))
            add_row("Backward MB (compiled)", fmt_mem("backward_memory", "compiled"))

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
    parser.add_argument("--warmup", type=int, default=5, help="Number of warmup iterations.")
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

    summaries: List[BenchmarkSummary] = []

    for name, config in configs.items():
        console.print(f"[cyan]Running eager benchmark for {name}[/cyan]")
        eager_metrics, param_stats, eager_oom = benchmark_config(
            name=name,
            config=config,
            device=device,
            seq_len=args.seq_length,
            batch_size=args.batch_size,
            warmup=args.warmup,
            steps=args.steps,
            do_compile=False,
        )
        compiled_metrics = None
        compiled_oom = False
        if not args.no_compile:
            console.print(f"[cyan]Running compiled benchmark for {name}[/cyan]")
            compiled_metrics, _, compiled_oom = benchmark_config(
                name=name,
                config=config,
                device=device,
                seq_len=args.seq_length,
                batch_size=args.batch_size,
                warmup=max(args.warmup, 1),
                steps=args.steps,
                do_compile=True,
            )

        summaries.append(
            BenchmarkSummary(
                name=name,
                config=config,
                param_stats=param_stats,
                eager_metrics=eager_metrics,
                compiled_metrics=compiled_metrics,
                eager_oom=eager_oom,
                compiled_oom=compiled_oom,
            )
        )

        if device.type == "cuda":
            torch.cuda.empty_cache()

    print_summary_table(summaries, device)


if __name__ == "__main__":
    main()
