#!/usr/bin/env python3
"""Benchmark configurable NanoGPT model providers with timing and memory stats.

This utility walks every public `CONFIG_*` configuration declared in the
selected model provider module and reports forward/backward timing plus memory
for both eager and `torch.compile` modes.  The script was written with the
following guard-rails in mind:

* Treat inference and training measurements separately so Transformer Engine
  kernels (which can require graph capture) do not leak across modes.
* Fail loudly but gracefully when `torch.compile` or the CUDA allocator hit an
  unsupported path—every OOM or compile failure is surfaced as "OOM" in the
  summary table rather than bubbling an exception.
* Capture a fused AdamW optimizer step so optimizer timing/memory is visible
  alongside pure forward/backward passes.
* Isolate inference, gradient, and optimizer stages: every run rebuilds its
  own model, data, and optimizer and scrubs CUDA state before and after
  measurement so torch.compile sees a pristine module.
* Keep GPU state tidy between configs: TF32 is enabled for performance,
  allocator segments are expanded to mitigate fragmentation, and caches are
  cleared after each run.

Future maintainers should skim `_torch_compile_model()` and
`benchmark_config()` first—they contain the subtleties around Dynamo
configuration and per-config cleanup.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import math
import statistics
import time
import warnings
from functools import lru_cache
from dataclasses import asdict, dataclass, is_dataclass
from types import ModuleType
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Mapping, Sequence
from datetime import datetime
from pathlib import Path
import re
import gc
import os

import torch
import torch.nn.functional as F
from rich.console import Console
from rich.table import Table
from torch.optim import Optimizer

from nanugpt.models import nanogpt as nanogpt_models, te_llama3
from nanugpt import utils


console = Console()

SUPPRESS_COMPILE_WARNINGS = False  # Set to False to see full torch.compile warning output.

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_RICH_TAG_PATTERN = re.compile(r"\[/?[^\]]+\]")


def _strip_markup(text: str) -> str:
    return _RICH_TAG_PATTERN.sub("", str(text))

MODEL_PROVIDERS: List[ModuleType] = [
    nanogpt_models,
    te_llama3,
]  # Update this list to change benchmarked providers.


def _provider_display_name(provider: ModuleType) -> str:
    return provider.__name__.split(".")[-1]


def _provider_names(providers: Optional[Sequence[ModuleType]] = None) -> List[str]:
    providers = providers if providers is not None else MODEL_PROVIDERS
    return [_provider_display_name(provider) for provider in providers]


def _resolve_output_directory() -> Path:
    raw_dir = os.environ.get("OUT_DIR")
    target_dir = raw_dir if raw_dir else "~/temp"
    expanded = os.path.expanduser(os.path.expandvars(target_dir))
    path = Path(expanded).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path

Metrics = Dict[str, float]
ParamStats = Dict[str, Optional[int]]


@dataclass(frozen=True)
class DiscoveredConfig:
    provider: ModuleType
    provider_display_name: str
    attr_name: str
    config: Any

    @property
    def qualified_name(self) -> str:
        return f"{self.provider_display_name}.{self.attr_name}"


def _is_cuda_device(device: torch.device) -> bool:
    """Return True when the target device can execute CUDA kernels."""
    return device.type == "cuda" and torch.cuda.is_available()


def _synchronize_cuda(device: torch.device) -> None:
    """Synchronize CUDA work if the target device is backed by CUDA."""
    if _is_cuda_device(device):
        torch.cuda.synchronize(device)


@dataclass
class ScenarioMetrics:
    """Per-stage measurements for eager and compiled modes."""
    eager: Optional[Metrics]
    compiled: Optional[Metrics]
    eager_oom: bool = False
    compiled_oom: bool = False
    supported: bool = True


@dataclass
class BenchmarkSummary:
    """Printable snapshot of a single config's parameters and timing results."""
    name: str
    provider_display_name: str
    config: Any
    param_stats: ParamStats
    stages: Dict[str, ScenarioMetrics]
    batch_size: int
    context_length: int
    tokens_per_step: int


def _torch_compile_model(
    model: torch.nn.Module,
    device: torch.device,
    provider: ModuleType,
) -> torch.nn.Module:
    """Wrap `torch.compile` with provider-friendly defaults (no cudagraph capture)."""
    if device.type != "cuda":
        raise RuntimeError(
            f"torch.compile() for provider {_provider_display_name(provider)} requires a CUDA device."
        )

    if SUPPRESS_COMPILE_WARNINGS:
        _configure_compile_environment()
    previous_cudagraphs = None
    try:
        import torch._dynamo as dynamo  # type: ignore[attr-defined]

        if hasattr(dynamo.config, "use_cudagraphs"):
            previous_cudagraphs = dynamo.config.use_cudagraphs
            dynamo.config.use_cudagraphs = False  # compiled inference prefers raw streams
        return torch.compile(model, mode="reduce-overhead", dynamic=True)
    finally:
        if previous_cudagraphs is not None:
            try:
                dynamo.config.use_cudagraphs = previous_cudagraphs  # type: ignore[name-defined]
            except Exception:
                pass

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


def configure_torch_runtime(device: torch.device) -> None:
    """Set global knobs (TF32, cudnn) for repeatable high-throughput runs."""
    torch.set_float32_matmul_precision("high")
    if _is_cuda_device(device):
        device_index = device.index if device.index is not None else 0
        torch.cuda.set_device(device_index)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def discover_configs(prefix: str = "CONFIG_") -> Dict[str, DiscoveredConfig]:
    """Discover dataclass configs in the configured model providers that match the prefix."""
    configs: Dict[str, DiscoveredConfig] = {}
    for provider in MODEL_PROVIDERS:
        provider_display = _provider_display_name(provider)
        for name in sorted(dir(provider)):
            if not name.startswith(prefix):
                continue
            value = getattr(provider, name)
            if is_dataclass(value) and not isinstance(value, type):
                entry = DiscoveredConfig(
                    provider=provider,
                    provider_display_name=provider_display,
                    attr_name=name,
                    config=value,
                )
                configs[entry.qualified_name] = entry
    return configs


def format_bytes_as_mb(num_bytes: float) -> float:
    """Convert bytes to megabytes."""
    return num_bytes / (1024 ** 2)


def mean_or_nan(values: Iterable[float]) -> float:
    """Mean that skips None/NaN entries so missing samples do not skew results."""
    filtered = [
        value
        for value in values
        if value is not None and not (isinstance(value, float) and math.isnan(value))
    ]
    return statistics.mean(filtered) if filtered else float("nan")


def aggregate_metrics(records: List[Metrics]) -> Metrics:
    """Aggregate timing and memory stats across iterations."""
    if not records:
        return {}
    keys = set().union(*(record.keys() for record in records))
    summary: Metrics = {}
    for key in keys:
        summary[key] = mean_or_nan(
            record[key] for record in records if key in record
        )
    return summary


def _cleanup_device_state(device: torch.device) -> None:
    """Clear cached memory and reset CUDA stats to isolate stages."""
    if _is_cuda_device(device):
        try:
            _synchronize_cuda(device)
        except (torch.cuda.CudaError, RuntimeError):
            pass
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
    gc.collect()


def _seed_everything(device: torch.device) -> None:
    """Re-seed all RNGs so each stage is repeatable and isolated."""
    torch.manual_seed(42)
    if _is_cuda_device(device):
        torch.cuda.manual_seed_all(42)


def _synthetic_loss(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, int]:
    # model_output: [batch_size, seq_len, vocab_size]
    # cross entropy loss expects a tensor of shape [batch_size, num_classes] and [batch_size]

    if isinstance(logits, Mapping):
        logits = logits['logits']

    preds = logits.view(-1, logits.size(-1)) # [batch_size*seq_len, vocab_size]
    targets = labels.view(-1) # [batch_size*seq_len]

    # ignore_index=-1 is actually not needed because we never output -ve index for tokens.
    # PyTorch default is -100. The negative index is used to ignore the loss for padding tokens.
    loss = F.cross_entropy(preds, targets, ignore_index=-1)
    # dim=-1 means we take the max along the last dimension, which is the vocab_size, so max is taken over the vocab
    correct = utils.safe_int_item((torch.argmax(preds, dim=-1) == targets).sum())

    return loss, correct # total num of predictions


def _prepare_inputs(
    config: Any,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    with_labels: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    context_length = min(seq_len, config.block_size)
    input_ids = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(batch_size, context_length),
        device=device,
        dtype=torch.long,
    )
    labels = None
    if with_labels:
        labels = torch.randint(
            low=0,
            high=config.vocab_size,
            size=(batch_size, context_length),
            device=device,
            dtype=torch.long,
        )
    return input_ids, labels


def _zero_grad(module: torch.nn.Module) -> None:
    try:
        module.zero_grad(set_to_none=True)
    except TypeError:
        module.zero_grad()


def _zero_optimizer(optimizer: Optimizer) -> None:
    try:
        optimizer.zero_grad(set_to_none=True)
    except TypeError:
        optimizer.zero_grad()


def _config_to_model_kwargs(config: Any) -> Dict[str, Any]:
    """Convert a provider config dataclass into kwargs accepted by `get_model`."""
    if is_dataclass(config):
        cfg_dict = asdict(config)
    else:
        cfg_dict = dict(vars(config))
    if "block_size" in cfg_dict and "context_length" not in cfg_dict:
        cfg_dict["context_length"] = cfg_dict["block_size"]
    return cfg_dict


def _build_model_from_provider(
    provider: ModuleType,
    config: Any,
    get_loss: Optional[Callable[..., Tuple[torch.Tensor, int]]],
) -> torch.nn.Module:
    """Instantiate a model using the configured provider's `get_model` entrypoint."""
    model_kwargs = _config_to_model_kwargs(config)
    get_model_fn = provider.get_model
    signature = inspect.signature(get_model_fn)
    constructed_kwargs: Dict[str, Any] = {}

    for name, param in signature.parameters.items():
        if name == "get_loss":
            constructed_kwargs[name] = get_loss
            continue
        if name in model_kwargs:
            constructed_kwargs[name] = model_kwargs[name]
            continue
        if param.default is inspect._empty:
            raise KeyError(
                f"Required parameter '{name}' is missing from config {config} for provider {provider.__name__}."
            )

    return get_model_fn(**constructed_kwargs)


def _embedding_parameter_count(model: torch.nn.Module) -> Optional[int]:
    """Best-effort retrieval of embedding parameter counts across providers."""
    embedding_module = getattr(model, "tok_embeddings", None)
    if embedding_module is None:
        transformer = getattr(model, "transformer", None)
        if transformer is not None:
            embedding_module = getattr(transformer, "wte", None)
    if embedding_module is None:
        return None
    weight = getattr(embedding_module, "weight", None)
    if weight is None:
        return None
    return weight.numel()


def _measure_inference_iteration(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    device: torch.device,
) -> Metrics:
    """Profile a single inference forward pass."""
    model.eval()
    is_cuda = _is_cuda_device(device)
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)
        mem_before = torch.cuda.memory_allocated(device)
        _synchronize_cuda(device)
    else:
        mem_before = 0.0

    start = time.perf_counter()
    with torch.no_grad():
        model(input_ids, return_logits=True)
    if is_cuda:
        _synchronize_cuda(device)
    forward_time = time.perf_counter() - start

    forward_memory = (
        torch.cuda.max_memory_allocated(device) - mem_before if is_cuda else float("nan")
    )

    return {
        "forward_time": forward_time,
        "forward_memory": float(forward_memory) if is_cuda else float("nan"),
    }


def _measure_training_iteration(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    optimizer: Optional[Optimizer] = None,
) -> Metrics:
    """Profile a training iteration with optional optimizer step."""
    model.train()
    is_cuda = _is_cuda_device(device)

    if optimizer is not None:
        _zero_optimizer(optimizer)
    _zero_grad(model)

    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)
        mem_before_forward = torch.cuda.memory_allocated(device)
        _synchronize_cuda(device)
    else:
        mem_before_forward = 0.0

    start_forward = time.perf_counter()
    _, loss, _ = model(input_ids, labels=labels, return_logits=False)
    if is_cuda:
        _synchronize_cuda(device)
    forward_time = time.perf_counter() - start_forward

    if is_cuda:
        forward_memory = torch.cuda.max_memory_allocated(device) - mem_before_forward
        torch.cuda.reset_peak_memory_stats(device)
        mem_before_backward = torch.cuda.memory_allocated(device)
    else:
        forward_memory = float("nan")
        mem_before_backward = 0.0

    start_backward = time.perf_counter()
    loss.backward()
    if is_cuda:
        _synchronize_cuda(device)
    backward_time = time.perf_counter() - start_backward

    if is_cuda:
        backward_memory = torch.cuda.max_memory_allocated(device) - mem_before_backward
    else:
        backward_memory = float("nan")

    metrics: Metrics = {
        "forward_time": float(forward_time),
        "forward_memory": float(forward_memory) if is_cuda else float("nan"),
        "backward_time": float(backward_time),
        "backward_memory": float(backward_memory) if is_cuda else float("nan"),
    }

    if optimizer is not None:
        if is_cuda:
            torch.cuda.reset_peak_memory_stats(device)
            mem_before_step = torch.cuda.memory_allocated(device)
            _synchronize_cuda(device)
        else:
            mem_before_step = 0.0

        start_step = time.perf_counter()
        optimizer.step()
        if is_cuda:
            _synchronize_cuda(device)
        step_time = time.perf_counter() - start_step

        step_memory = (
            torch.cuda.max_memory_allocated(device) - mem_before_step
            if is_cuda
            else float("nan")
        )

        metrics.update(
            {
                "optimizer_step_time": float(step_time),
                "optimizer_step_memory": float(step_memory) if is_cuda else float("nan"),
            }
        )
        _zero_optimizer(optimizer)

    iteration_time = metrics["forward_time"] + metrics["backward_time"]
    iteration_memory = (
        max(metrics["forward_memory"], metrics["backward_memory"])
        if is_cuda
        else float("nan")
    )
    if optimizer is not None:
        iteration_time += metrics["optimizer_step_time"]
        if is_cuda:
            iteration_memory = max(iteration_memory, metrics["optimizer_step_memory"])

    metrics["iteration_time"] = float(iteration_time)
    metrics["iteration_memory"] = float(iteration_memory)

    _zero_grad(model)
    return metrics


def _build_fused_adamw(model: torch.nn.Module, device: torch.device) -> Optional[Optimizer]:
    """Instantiate a fused AdamW optimizer if the runtime supports it."""
    if not _is_cuda_device(device):
        return None
    try:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            betas=(0.9, 0.95),
            weight_decay=0.0,
            fused=True,
        )
    except (TypeError, RuntimeError, ValueError):
        return None
    return optimizer


@dataclass
class StageRunResult:
    metrics: Optional[Metrics]
    oom: bool
    supported: bool = True


INFERENCE_STAGE = "inference"
TRAINING_STAGE = "training"
OPTIMIZER_STAGE = "training_adamw"


def _stage_label(stage: str) -> str:
    return {
        INFERENCE_STAGE: "inference",
        TRAINING_STAGE: "training",
        OPTIMIZER_STAGE: "training (AdamW)",
    }.get(stage, stage)


def _run_stage_once(
    config_name: str,
    config: Any,
    provider: ModuleType,
    device: torch.device,
    batch_size: int,
    seq_len: int,
    warmup: int,
    steps: int,
    stage: str,
    compile_model: bool,
) -> StageRunResult:
    """Run a single stage (optionally compiled) and collect aggregated metrics."""
    _cleanup_device_state(device)
    _seed_everything(device)

    requires_labels = stage != INFERENCE_STAGE
    requires_optimizer = stage == OPTIMIZER_STAGE
    stage_label = _stage_label(stage)

    try:
        model = _build_model_from_provider(
            provider=provider,
            config=config,
            get_loss=_synthetic_loss if requires_labels else None,
        )
    except torch.cuda.OutOfMemoryError:
        console.print(
            f"[bold red]{config_name}: OOM creating {stage_label} model ({'compiled' if compile_model else 'eager'}).[/bold red]"
        )
        return StageRunResult(metrics=None, oom=True)

    try:
        model = model.to(device)
    except torch.cuda.OutOfMemoryError:
        console.print(
            f"[bold red]{config_name}: OOM moving {stage_label} model to {device} ({'compiled' if compile_model else 'eager'}).[/bold red]"
        )
        del model
        return StageRunResult(metrics=None, oom=True)

    optimizer: Optional[Optimizer] = None
    try:
        if compile_model:
            console.print(
                f"[dim]{config_name}: compiling {stage_label} model with torch.compile...[/dim]"
            )
            if stage == INFERENCE_STAGE:
                model.eval()
            else:
                model.train()
            try:
                model = _torch_compile_model(model=model, device=device, provider=provider)
            except torch.cuda.OutOfMemoryError:
                console.print(
                    f"[bold red]{config_name}: OOM compiling {stage_label} model.[/bold red]"
                )
                return StageRunResult(metrics=None, oom=True)
            except Exception as exc:
                console.print(
                    f"[bold red]{config_name}: torch.compile failed for {stage_label} ({exc.__class__.__name__}: {exc}).[/bold red]"
                )
                return StageRunResult(metrics=None, oom=True)
            if _is_cuda_device(device):
                torch.cuda.reset_peak_memory_stats(device)
                _synchronize_cuda(device)

        if requires_optimizer:
            optimizer = _build_fused_adamw(model, device)
            if optimizer is None:
                console.print(
                    f"[bold yellow]{config_name}: fused AdamW not available; skipping {stage_label} measurements.[/bold yellow]"
                )
                return StageRunResult(metrics=None, oom=False, supported=False)

        try:
            input_ids, labels = _prepare_inputs(
                config,
                batch_size=batch_size,
                seq_len=seq_len,
                device=device,
                with_labels=requires_labels,
            )
        except torch.cuda.OutOfMemoryError:
            console.print(
                f"[bold red]{config_name}: OOM generating synthetic data for {stage_label} ({'compiled' if compile_model else 'eager'}).[/bold red]"
            )
            return StageRunResult(metrics=None, oom=True)

        for i in range(max(warmup, 0)):
            console.print(
                f"[dim]{config_name}: {stage_label} warmup {i + 1}/{warmup} ({'compiled' if compile_model else 'eager'})[/dim]"
            )
            try:
                if stage == INFERENCE_STAGE:
                    _measure_inference_iteration(model, input_ids, device)
                else:
                    assert labels is not None
                    _measure_training_iteration(
                        model, input_ids, labels, device, optimizer=optimizer
                    )
            except torch.cuda.OutOfMemoryError:
                console.print(
                    f"[bold red]{config_name}: OOM during {stage_label} warmup ({'compiled' if compile_model else 'eager'}).[/bold red]"
                )
                return StageRunResult(metrics=None, oom=True)

        records: List[Metrics] = []
        for i in range(max(steps, 0)):
            console.print(
                f"[dim]{config_name}: {stage_label} step {i + 1}/{steps} ({'compiled' if compile_model else 'eager'})[/dim]"
            )
            try:
                if stage == INFERENCE_STAGE:
                    metrics = _measure_inference_iteration(model, input_ids, device)
                else:
                    assert labels is not None
                    metrics = _measure_training_iteration(
                        model, input_ids, labels, device, optimizer=optimizer
                    )
            except torch.cuda.OutOfMemoryError:
                console.print(
                    f"[bold red]{config_name}: OOM during {stage_label} measurement ({'compiled' if compile_model else 'eager'}).[/bold red]"
                )
                return StageRunResult(metrics=None, oom=True)
            records.append(metrics)

        aggregated = aggregate_metrics(records)
        return StageRunResult(metrics=aggregated, oom=False)

    finally:
        if optimizer is not None:
            del optimizer
        del model
        _cleanup_device_state(device)


def benchmark_config(
    name: str,
    provider: ModuleType,
    config: Any,
    device: torch.device,
    seq_len: int,
    batch_size: int,
    warmup: int,
    steps: int,
    run_compiled: bool,
) -> Tuple[Dict[str, ScenarioMetrics], ParamStats]:
    """Benchmark all stages for a configuration, optionally including torch.compile."""
    param_stats: ParamStats = {"total": None, "trainable": None, "embedding": None}

    try:
        reference_model = _build_model_from_provider(
            provider=provider,
            config=config,
            get_loss=_synthetic_loss,
        )
        param_stats["total"] = sum(p.numel() for p in reference_model.parameters())
        param_stats["trainable"] = sum(
            p.numel() for p in reference_model.parameters() if p.requires_grad
        )
        embedding_params = _embedding_parameter_count(reference_model)
        if embedding_params is not None:
            param_stats["embedding"] = embedding_params
    except Exception:
        console.print(
            f"[bold yellow]{name}: Unable to compute parameter statistics for this configuration.[/bold yellow]"
        )
    finally:
        try:
            del reference_model
        except UnboundLocalError:
            pass
        gc.collect()

    stages: Dict[str, ScenarioMetrics] = {}
    for stage in (INFERENCE_STAGE, TRAINING_STAGE, OPTIMIZER_STAGE):
        eager_result = _run_stage_once(
            config_name=name,
            config=config,
            provider=provider,
            device=device,
            batch_size=batch_size,
            seq_len=seq_len,
            warmup=warmup,
            steps=steps,
            stage=stage,
            compile_model=False,
        )

        if eager_result.oom and stage != OPTIMIZER_STAGE:
            stages[stage] = ScenarioMetrics(
                eager=None,
                compiled=None,
                eager_oom=True,
                compiled_oom=False,
                supported=True,
            )
            continue

        if not eager_result.supported:
            stages[stage] = ScenarioMetrics(
                eager=None,
                compiled=None,
                eager_oom=eager_result.oom,
                compiled_oom=False,
                supported=False,
            )
            continue

        compiled_result = StageRunResult(metrics=None, oom=False, supported=True)
        if run_compiled and not eager_result.oom:
            compiled_result = _run_stage_once(
                config_name=name,
                config=config,
                provider=provider,
                device=device,
                batch_size=batch_size,
                seq_len=seq_len,
                warmup=max(warmup, 1),
                steps=steps,
                stage=stage,
                compile_model=True,
            )

        stages[stage] = ScenarioMetrics(
            eager=eager_result.metrics,
            compiled=compiled_result.metrics if run_compiled else None,
            eager_oom=eager_result.oom,
            compiled_oom=compiled_result.oom if run_compiled else False,
            supported=eager_result.supported and (compiled_result.supported or not run_compiled),
        )

    return stages, param_stats


SCENARIO_SPECS = {
    INFERENCE_STAGE: {
        "label": "Inference",
        "color": "cyan",
        "time_rows": [("Forward ms", "forward_time")],
        "memory_rows": [("Forward MB", "forward_memory")],
    },
    TRAINING_STAGE: {
        "label": "Training (Grad)",
        "color": "magenta",
        "time_rows": [
            ("Forward ms", "forward_time"),
            ("Backward ms", "backward_time"),
            ("Iteration ms", "iteration_time"),
        ],
        "memory_rows": [
            ("Forward MB", "forward_memory"),
            ("Backward MB", "backward_memory"),
            ("Iteration MB", "iteration_memory"),
        ],
    },
    OPTIMIZER_STAGE: {
        "label": "Training + AdamW",
        "color": "green",
        "time_rows": [
            ("Forward ms", "forward_time"),
            ("Backward ms", "backward_time"),
            ("Optimizer Step ms", "optimizer_step_time"),
            ("Iteration ms", "iteration_time"),
        ],
        "memory_rows": [
            ("Forward MB", "forward_memory"),
            ("Backward MB", "backward_memory"),
            ("Optimizer Step MB", "optimizer_step_memory"),
            ("Iteration MB", "iteration_memory"),
        ],
    },
}

TEN_B_TOKENS = 10_000_000_000


def print_summary_table(
    results: List[BenchmarkSummary], device: torch.device
) -> Tuple[List[str], List[List[str]]]:
    """Render a Rich table with comparable stats across all requested configs and return tabular data."""
    if not results:
        console.print("[bold red]No benchmark results to display.[/bold red]")
        return [], []

    device_is_cuda = _is_cuda_device(device)
    provider_labels: List[str] = []
    for summary in results:
        if summary.provider_display_name not in provider_labels:
            provider_labels.append(summary.provider_display_name)
    title_prefix = ", ".join(provider_labels) if provider_labels else "Model"
    table = Table(title=f"{title_prefix} Benchmark Summary", expand=True, show_lines=True)
    table.add_column("Metric", justify="left", style="bold")
    for summary in results:
        table.add_column(summary.name, justify="right")

    headers = ["Metric"] + [summary.name for summary in results]
    table_rows: List[List[str]] = []

    def add_row(label: str, extractor: Callable[[BenchmarkSummary], str]) -> None:
        values = [extractor(summary) for summary in results]
        table.add_row(label, *values)
        table_rows.append(
            [_strip_markup(label)] + [_strip_markup(value) for value in values]
        )

    def add_group_header(label: str, color: str) -> None:
        markup_label = f"[bold {color}]{label}[/bold {color}]"
        blanks = ["" for _ in results]
        table.add_row(markup_label, *blanks)
        table_rows.append([label, *blanks])

    def fmt_config(attr: str) -> Callable[[BenchmarkSummary], str]:
        def _formatter(summary: BenchmarkSummary) -> str:
            value = getattr(summary.config, attr, None)
            if value is None:
                return "n/a"
            return str(value)

        return _formatter

    def fmt_params(key: str) -> Callable[[BenchmarkSummary], str]:
        def _formatter(summary: BenchmarkSummary) -> str:
            value = summary.param_stats.get(key)
            if value is None:
                return "n/a"
            return f"{float(value) / 1e6:.2f}M"

        return _formatter

    def fmt_stage_supported(stage: str) -> Callable[[BenchmarkSummary], str]:
        def _formatter(summary: BenchmarkSummary) -> str:
            stage_metrics = summary.stages.get(stage)
            if not stage_metrics or not stage_metrics.supported:
                return "[dim]No[/dim]"
            if (
                stage_metrics.eager is None
                and stage_metrics.eager_oom
                and not scenario_has_compiled(stage)
            ):
                return "[bold red]OOM[/bold red]"
            return "[green]Yes[/green]"

        return _formatter

    def fmt_tokens_per_step(summary: BenchmarkSummary) -> str:
        value = summary.tokens_per_step
        if value <= 0:
            return "n/a"
        return f"[bold]{value:,}[/bold]"

    def scenario_has_compiled(stage: str) -> bool:
        for summary in results:
            stage_metrics = summary.stages.get(stage)
            if not stage_metrics:
                continue
            if stage_metrics.compiled is not None or stage_metrics.compiled_oom:
                return True
        return False

    def fmt_time_for_tokens(stage: str, mode: str) -> Callable[[BenchmarkSummary], str]:
        def _formatter(summary: BenchmarkSummary) -> str:
            stage_metrics = summary.stages.get(stage)
            if not stage_metrics or not stage_metrics.supported:
                return "n/a"
            if summary.tokens_per_step <= 0:
                return "n/a"
            if mode == "compiled" and not scenario_has_compiled(stage):
                return "n/a"
            metrics = stage_metrics.eager if mode == "eager" else stage_metrics.compiled
            oom = stage_metrics.eager_oom if mode == "eager" else stage_metrics.compiled_oom
            if oom:
                return "OOM"
            if not metrics or "iteration_time" not in metrics:
                return "n/a"
            iteration_time = metrics["iteration_time"]
            if iteration_time is None or (isinstance(iteration_time, float) and math.isnan(iteration_time)):
                return "n/a"
            seconds = (TEN_B_TOKENS / summary.tokens_per_step) * iteration_time
            hours = seconds / 3600.0
            return f"[bold]{hours:,.2f}h[/bold]"

        return _formatter

    def extract_metric(summary: BenchmarkSummary, stage: str, mode: str, key: str) -> str:
        stage_metrics = summary.stages.get(stage)
        if not stage_metrics or not stage_metrics.supported:
            return "n/a"
        metrics = stage_metrics.eager if mode == "eager" else stage_metrics.compiled
        oom = stage_metrics.eager_oom if mode == "eager" else stage_metrics.compiled_oom
        if mode == "compiled" and not scenario_has_compiled(stage):
            return "n/a"
        if oom:
            return "OOM"
        if not metrics or key not in metrics:
            return "n/a"
        value = metrics[key]
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return "n/a"
        if key.endswith("_time"):
            return f"{value * 1e3:,.2f}"
        if key.endswith("_memory"):
            return (
                f"{format_bytes_as_mb(value):,.2f}"
                if device_is_cuda
                else "n/a"
            )
        return f"{value:,.4f}"

    add_group_header("Configuration", "bright_blue")
    add_row("n_embd", fmt_config("n_embd"))
    add_row("n_layer", fmt_config("n_layer"))
    add_row("n_head", fmt_config("n_head"))
    add_row("n_kv_heads", fmt_config("n_kv_heads"))
    add_row("block_size", fmt_config("block_size"))
    add_row("vocab_size", fmt_config("vocab_size"))
    add_row("Batch Size", lambda summary: str(summary.batch_size))
    add_row("Fused AdamW", fmt_stage_supported(OPTIMIZER_STAGE))

    add_group_header("Parameter Counts", "bright_blue")
    add_row("Embedding Params (M)", fmt_params("embedding"))
    add_row("Trainable Params (M)", fmt_params("trainable"))
    add_row("Total Params (M)", fmt_params("total"))

    for stage, spec in SCENARIO_SPECS.items():
        add_group_header(spec["label"], spec["color"])
        add_row(
            "Supported",
            fmt_stage_supported(stage),
        )

        compiled_available = scenario_has_compiled(stage)

        add_row(
            "Forward ms (eager)",
            lambda summary, stage=stage: extract_metric(summary, stage, "eager", "forward_time"),
        )
        if compiled_available:
            add_row(
                "Forward ms (compiled)",
                lambda summary, stage=stage: extract_metric(summary, stage, "compiled", "forward_time"),
            )

        if stage != INFERENCE_STAGE:
            add_row(
                "Backward ms (eager)",
                lambda summary, stage=stage: extract_metric(summary, stage, "eager", "backward_time"),
            )
            if compiled_available:
                add_row(
                    "Backward ms (compiled)",
                    lambda summary, stage=stage: extract_metric(summary, stage, "compiled", "backward_time"),
                )

        if stage == OPTIMIZER_STAGE:
            add_row(
                "Optimizer Step ms (eager)",
                lambda summary, stage=stage: extract_metric(
                    summary, stage, "eager", "optimizer_step_time"
                ),
            )
            if compiled_available:
                add_row(
                    "Optimizer Step ms (compiled)",
                    lambda summary, stage=stage: extract_metric(
                        summary, stage, "compiled", "optimizer_step_time"
                    ),
                )

        if stage != INFERENCE_STAGE:
            add_row(
                "Iteration ms (eager)",
                lambda summary, stage=stage: extract_metric(summary, stage, "eager", "iteration_time"),
            )
            if compiled_available:
                add_row(
                    "Iteration ms (compiled)",
                    lambda summary, stage=stage: extract_metric(
                        summary, stage, "compiled", "iteration_time"
                    ),
                )
            if stage in (TRAINING_STAGE, OPTIMIZER_STAGE):
                add_row(
                    "Tokens / step",
                    lambda summary, stage=stage: (
                        fmt_tokens_per_step(summary)
                        if (stage_metrics := summary.stages.get(stage)) and stage_metrics.supported
                        else "n/a"
                    ),
                )
                add_row(
                    "10B tokens h (eager)",
                    fmt_time_for_tokens(stage, "eager"),
                )
                if compiled_available:
                    add_row(
                        "10B tokens h (compiled)",
                        fmt_time_for_tokens(stage, "compiled"),
                    )

        if device_is_cuda:
            add_row(
                "Forward MB (eager)",
                lambda summary, stage=stage: extract_metric(summary, stage, "eager", "forward_memory"),
            )
            if compiled_available:
                add_row(
                    "Forward MB (compiled)",
                    lambda summary, stage=stage: extract_metric(summary, stage, "compiled", "forward_memory"),
                )

            if stage != INFERENCE_STAGE:
                add_row(
                    "Backward MB (eager)",
                    lambda summary, stage=stage: extract_metric(summary, stage, "eager", "backward_memory"),
                )
                if compiled_available:
                    add_row(
                        "Backward MB (compiled)",
                        lambda summary, stage=stage: extract_metric(summary, stage, "compiled", "backward_memory"),
                    )

            if stage == OPTIMIZER_STAGE:
                add_row(
                    "Optimizer Step MB (eager)",
                    lambda summary, stage=stage: extract_metric(
                        summary, stage, "eager", "optimizer_step_memory"
                    ),
                )
                if compiled_available:
                    add_row(
                        "Optimizer Step MB (compiled)",
                        lambda summary, stage=stage: extract_metric(
                            summary, stage, "compiled", "optimizer_step_memory"
                        ),
                    )

            if stage != INFERENCE_STAGE:
                add_row(
                    "Iteration MB (eager)",
                    lambda summary, stage=stage: extract_metric(
                        summary, stage, "eager", "iteration_memory"
                    ),
                )
                if compiled_available:
                    add_row(
                        "Iteration MB (compiled)",
                        lambda summary, stage=stage: extract_metric(
                            summary, stage, "compiled", "iteration_memory"
                        ),
                    )

    console.print(table)
    return headers, table_rows


def write_summary_csv(headers: List[str], rows: List[List[str]]) -> Path:
    """Persist the rendered summary table to a CSV file and return its path."""
    output_dir = _resolve_output_directory()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"benchmark_summary_{timestamp}.csv"
    output_path = (output_dir / filename).resolve()
    with output_path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(rows)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Benchmark {', '.join(_provider_names())} configs with timing and memory stats."
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for synthetic data.")
    parser.add_argument(
        "--seq-length",
        type=int,
        default=512,
        help="Sequence length for synthetic data (capped at config block_size).",
    )
    parser.add_argument("--warmup", type=int, default=2, help="Number of warmup iterations.")
    parser.add_argument("--steps", type=int, default=1, help="Number of timed iterations.")
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
        help="Run a single config (CONFIG_* attribute or provider-qualified like te_llama3.CONFIG_124M).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    provider_names = _provider_names()
    provider_label = ", ".join(provider_names)
    provider_plural = "provider" if len(provider_names) == 1 else "providers"

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but not available on this system.")

    configure_torch_runtime(device)
    is_cuda = _is_cuda_device(device)

    if not is_cuda:
        console.print(
            f"[bold yellow]CUDA device not detected. {provider_label} {provider_plural} generally require GPU support for benchmarking.[/bold yellow]"
        )

    run_compiled = not args.no_compile
    if run_compiled and not hasattr(torch, "compile"):
        console.print(
            "[bold yellow]torch.compile is unavailable in this PyTorch build; skipping compiled benchmarks.[/bold yellow]"
        )
        run_compiled = False
    if run_compiled and not is_cuda:
        console.print(
            f"[bold yellow]Skipping compiled benchmarks because {provider_label} compilation requires a CUDA-capable device.[/bold yellow]"
        )
        run_compiled = False

    configs = discover_configs()
    config_items = list(configs.items())

    if args.config:
        filtered_items = [
            (name, entry)
            for name, entry in config_items
            if name == args.config or entry.attr_name == args.config
        ]
        if not filtered_items:
            available_configs = ", ".join(sorted(configs))
            raise ValueError(
                f"Unknown config '{args.config}'. Available: {available_configs}"
            )
        config_items = filtered_items

    if not config_items:
        console.print(
            f"[bold red]No CONFIG_ prefixed dataclass configs found in providers {provider_label}.[/bold red]"
        )
        return

    summaries: List[BenchmarkSummary] = []

    for name, entry in config_items:
        config = entry.config
        # scrub allocator state so each config starts from the same baseline
        if is_cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            _synchronize_cuda(device)

        console.print(f"[cyan]Benchmarking {name}[/cyan]")
        stages, param_stats = benchmark_config(
            name=name,
            provider=entry.provider,
            config=config,
            device=device,
            seq_len=args.seq_length,
            batch_size=args.batch_size,
            warmup=args.warmup,
            steps=args.steps,
            run_compiled=run_compiled,
        )
        summaries.append(
            BenchmarkSummary(
                name=name,
                provider_display_name=entry.provider_display_name,
                config=config,
                param_stats=param_stats,
                stages=stages,
                batch_size=args.batch_size,
                context_length=min(args.seq_length, getattr(config, "block_size", args.seq_length)),
                tokens_per_step=args.batch_size
                * min(args.seq_length, getattr(config, "block_size", args.seq_length)),
            )
        )

    headers, table_rows = print_summary_table(summaries, device)
    if table_rows:
        output_path = write_summary_csv(headers, table_rows)
        console.print(f"[bold green]Benchmark summary CSV saved to {output_path}[/bold green]")


if __name__ == "__main__":
    main()
