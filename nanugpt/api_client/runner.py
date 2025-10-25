from __future__ import annotations

"""
High-level prompt execution runner that coordinates AOAIClient requests.

The module provides `run()` which:
  - Normalizes PromptRequest sequences into deterministic worker slices so resumes are possible.
  - Streams progress to a Rich dashboard (per-worker status + aggregated stats).
  - Writes per-worker JSONL shards during execution, then merges them into `responses.jsonl`.
  - Persists run metadata (`run_meta.json`) to track how many times a run has been resumed.
Resumption is transparent: we read existing JSONL shards, reconstruct PromptResult objects, and
continue from the exact prompt where each worker left off. This keeps final outputs identical
between uninterrupted runs and runs with interruptions + resume.
"""

import asyncio
import concurrent.futures
import json
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from concurrent.futures import ThreadPoolExecutor

import yaml
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from .aoai_client import AOAIClient, AzureConfig, ChatResult
from .prompt_exec import (
    PromptExecutor,
    PromptRequest,
    PromptResult,
    CompleteHook,
    ErrorHook,
    RetryHook,
    StartHook,
)


@dataclass
class DashboardState:
    total: int
    lock: threading.Lock = field(default_factory=threading.Lock)
    started: int = 0
    completed: int = 0
    failed: int = 0
    retries: int = 0
    worker_status: Dict[str, str] = field(default_factory=dict)
    start_time: float = field(default_factory=time.perf_counter)
    resumes: int = 0

    def snapshot(self) -> Dict[str, object]:
        with self.lock:
            return {
                "total": self.total,
                "started": self.started,
                "completed": self.completed,
                "failed": self.failed,
                "retries": self.retries,
                "workers": dict(self.worker_status),
                "elapsed": time.perf_counter() - self.start_time,
                "resumes": self.resumes,
            }


@dataclass(frozen=True)
class ExecutorHooks:
    on_start: Optional[StartHook] = None
    on_complete: Optional[CompleteHook] = None
    on_error: Optional[ErrorHook] = None
    on_retry: Optional[RetryHook] = None


def _call_hook(hook: Optional[Callable[..., Any]], *args: Any) -> None:
    """Execute hook callables, awaiting async coroutines when necessary."""
    if hook is None:
        return
    outcome = hook(*args)
    if asyncio.iscoroutine(outcome):
        asyncio.run(outcome)


def _build_dashboard(snapshot: Dict[str, object]) -> Panel:
    total = snapshot["total"]
    completed = snapshot["completed"]
    failed = snapshot["failed"]
    started = snapshot["started"]
    retries = snapshot["retries"]
    elapsed = snapshot["elapsed"]
    in_progress = started - (completed + failed)
    pct = (completed + failed) / total * 100 if total else 0.0

    summary = Table.grid(padding=(0, 1))
    summary.add_column("Metric", justify="right", style="cyan", no_wrap=True)
    summary.add_column("Value", style="bold")
    summary.add_row("Total", str(total))
    summary.add_row("Started", str(started))
    summary.add_row("Completed", str(completed))
    summary.add_row("Failed", str(failed))
    summary.add_row("In Progress", str(in_progress))
    summary.add_row("Retries", str(retries))
    summary.add_row("Resumes", str(snapshot.get("resumes", 0)))
    summary.add_row("Elapsed (s)", f"{elapsed:0.1f}")
    summary.add_row("Progress", f"{pct:0.1f}%")

    worker_table = Table(title="Workers", box=None)
    worker_table.add_column("Worker", style="green")
    worker_table.add_column("Status")
    workers: Dict[str, str] = snapshot["workers"]  # type: ignore[assignment]
    if workers:
        for name, status in sorted(workers.items()):
            worker_table.add_row(name, status)
    else:
        worker_table.add_row("—", "Waiting for work")

    return Panel(Group(summary, worker_table), title="Prompt Execution Dashboard", border_style="magenta")


def _ensure_output_dir(
    base_output_dir: Optional[Path],
    *,
    subdir: str = "prompt_entropy",
) -> Path:
    """Create the output directory (or reuse it) honoring env and CLI overrides."""
    out_env = os.environ.get("OUT_DIR")
    base = base_output_dir or (Path(out_env) if out_env else Path.cwd())
    target = base / subdir
    target.mkdir(parents=True, exist_ok=True)
    return target


def _result_to_record(result: PromptResult) -> Dict[str, Any]:
    metadata = result.request.metadata
    request = result.request
    return {
        "id": metadata.get("id"),
        "metadata": metadata,
        "status": "succeeded" if result.succeeded else "failed",
        "response": result.chat_result.content if result.chat_result else None,
        "system_prompt": request.system_prompt,
        "user_prompt": request.user_prompt,
        "error": str(result.error) if result.error else None,
        "api_duration": result.api_duration,
        "retry_duration": result.retry_duration,
        "total_calls": result.total_calls,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "total_tokens": result.total_tokens,
        "started_at": result.started_at,
        "ended_at": result.ended_at,
    }


def _record_to_prompt_result(
    record: Dict[str, Any],
    request: PromptRequest,
    index: int,
) -> PromptResult:
    """Rehydrate a `PromptResult` from a JSONL record on disk."""
    chat_result: Optional[ChatResult] = None
    status = record.get("status")
    if status == "succeeded":
        chat_result = ChatResult(
            content=record.get("response") or "",
            input_tokens=int(record.get("input_tokens") or 0),
            output_tokens=int(record.get("output_tokens") or 0),
            total_tokens=int(record.get("total_tokens") or 0),
        )
    error: Optional[BaseException] = None
    if status != "succeeded":
        err_text = record.get("error")
        error = RuntimeError(err_text) if err_text else RuntimeError("Prompt failed in previous run")

    total_calls = int(record.get("total_calls") or 0)
    result = PromptResult(
        index=index,
        request=request,
        chat_result=chat_result,
        error=error,
        attempts=total_calls,
        started_at=float(record.get("started_at") or 0.0),
        ended_at=float(record.get("ended_at") or 0.0),
        api_duration=float(record.get("api_duration") or 0.0),
        retry_duration=float(record.get("retry_duration") or 0.0),
        input_tokens=int(record.get("input_tokens") or 0),
        output_tokens=int(record.get("output_tokens") or 0),
        total_tokens=int(record.get("total_tokens") or 0),
        total_calls=total_calls,
    )
    return result


def _write_stats(
    path: Path,
    results: Sequence[PromptResult],
    *,
    run_started_at: float,
    run_ended_at: float,
    resumes: int,
) -> None:
    """Persist aggregate run statistics for later analysis."""
    successes = sum(1 for r in results if r.succeeded)
    failures = len(results) - successes
    stats = {
        "total_prompts": len(results),
        "successes": successes,
        "failures": failures,
        "total_calls": sum(r.total_calls for r in results),
        "total_api_time_seconds": sum(r.api_duration for r in results),
        "total_retry_time_seconds": sum(r.retry_duration for r in results),
        "total_input_tokens": sum(r.input_tokens for r in results),
        "total_output_tokens": sum(r.output_tokens for r in results),
        "total_tokens": sum(r.total_tokens for r in results),
        "run_started_at": datetime.fromtimestamp(run_started_at, tz=timezone.utc).isoformat(),
        "run_ended_at": datetime.fromtimestamp(run_ended_at, tz=timezone.utc).isoformat(),
        "run_elapsed_seconds": run_ended_at - run_started_at,
        "average_api_time_seconds": (
            sum(r.api_duration for r in results) / len(results) if results else 0.0
        ),
        "average_total_calls": (
            sum(r.total_calls for r in results) / len(results) if results else 0.0
        ),
        "resumes": resumes,
    }
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(stats, f, sort_keys=False)


def _run_executor(
    client: AOAIClient,
    assignments: Sequence[Sequence[Tuple[int, PromptRequest]]],
    pending_assignments: Sequence[Sequence[Tuple[int, PromptRequest]]],
    pending_prefix_counts: Sequence[Sequence[int]],
    state: DashboardState,
    console: Console,
    worker_paths: Dict[str, Path],
    executor_hooks: ExecutorHooks,
) -> List[PromptResult]:
    """Run pending prompt slices per worker while keeping the dashboard in sync."""
    worker_count = len(assignments)
    new_results: List[PromptResult] = []
    results_lock = threading.Lock()
    user_on_start = executor_hooks.on_start
    user_on_complete = executor_hooks.on_complete
    user_on_error = executor_hooks.on_error
    user_on_retry = executor_hooks.on_retry

    def _format_status(prefix: str, metadata: Dict[str, object], extra: Optional[str] = None) -> str:
        parts = [prefix]
        if "id" in metadata:
            parts.append(f"id={metadata['id']}")
        if "rnd" in metadata:
            parts.append(f"rnd={metadata['rnd']}")
        if extra:
            parts.append(extra)
        return " ".join(parts)

    def _worker_loop(worker_id: int) -> List[PromptResult]:
        worker_name = f"worker_{worker_id:02d}"
        assigned_pairs = assignments[worker_id]
        pending_plan = list(pending_assignments[worker_id])
        prefix_counts = list(pending_prefix_counts[worker_id])
        if len(pending_plan) != len(prefix_counts):
            raise RuntimeError("PromptRunner internal mismatch: prefix metadata does not align with pending prompts")

        total_assigned = len(assigned_pairs)
        if not pending_plan:
            with state.lock:
                state.worker_status[worker_name] = "Resume complete" if total_assigned else "Idle"
            return []

        worker_executor = PromptExecutor(client, max_concurrency=1)

        def _global_index(local_index: int) -> int:
            return pending_plan[local_index][0]

        def _prefix_completed(local_index: int) -> int:
            return prefix_counts[local_index]

        def _progress_label(local_index: int) -> str:
            completed_so_far = _prefix_completed(local_index) + local_index + 1
            return f"{completed_so_far}/{total_assigned}"

        def _append_record(local_index: int, result: PromptResult) -> None:
            global_idx = _global_index(local_index)
            result.index = global_idx
            record = _result_to_record(result)
            record["index"] = global_idx
            record["worker"] = worker_name
            path = worker_paths[worker_name]
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        def on_start(local_index: int, request: PromptRequest, result: PromptResult) -> None:
            result.index = _global_index(local_index)
            _call_hook(user_on_start, local_index, request, result)
            with state.lock:
                state.started += 1
                state.worker_status[worker_name] = _format_status("Running", request.metadata, _progress_label(local_index))

        def on_complete(local_index: int, request: PromptRequest, chat_result: ChatResult, result: PromptResult) -> None:
            result.index = _global_index(local_index)
            _call_hook(user_on_complete, local_index, request, chat_result, result)
            _append_record(local_index, result)
            with state.lock:
                state.completed += 1
                state.worker_status[worker_name] = _format_status("Done", request.metadata, _progress_label(local_index))

        def on_error(local_index: int, request: PromptRequest, exc: BaseException, result: PromptResult) -> None:
            result.index = _global_index(local_index)
            _call_hook(user_on_error, local_index, request, exc, result)
            _append_record(local_index, result)
            with state.lock:
                state.failed += 1
                state.worker_status[worker_name] = _format_status(
                    "Error",
                    request.metadata,
                    f"{_progress_label(local_index)} {exc}",
                )

        def on_retry(local_index: int, request: PromptRequest, attempt: int, exc: Exception) -> None:
            _call_hook(user_on_retry, local_index, request, attempt, exc)
            with state.lock:
                state.retries += 1
                state.worker_status[worker_name] = _format_status(
                    "Retry",
                    request.metadata,
                    f"{_progress_label(local_index)} #{attempt}",
                )

        pending_requests = [req for _, req in pending_plan]
        results = worker_executor.run_prompts(
            pending_requests,
            concurrency=1,
            on_start=on_start,
            on_complete=on_complete,
            on_error=on_error,
            on_retry=on_retry,
        )
        for idx, result in enumerate(results):
            result.index = _global_index(idx)
        with state.lock:
            state.worker_status[worker_name] = _format_status("Done", pending_plan[-1][1].metadata, f"{total_assigned}/{total_assigned}")
        return results

    with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="PromptRunner") as pool:
        futures = {pool.submit(_worker_loop, worker_id): worker_id for worker_id in range(worker_count)}
        with Live(_build_dashboard(state.snapshot()), console=console, refresh_per_second=5) as live:
            while futures:
                done, _ = concurrent.futures.wait(
                    list(futures.keys()), timeout=0.2, return_when=concurrent.futures.FIRST_COMPLETED
                )
                live.update(_build_dashboard(state.snapshot()))
                for fut in done:
                    worker_id = futures.pop(fut)
                    worker_results = fut.result()
                    with results_lock:
                        new_results.extend(worker_results)
            live.update(_build_dashboard(state.snapshot()))

    return new_results


def run(
    requests: Sequence[PromptRequest],
    *,
    workers: int,
    base_output_dir: Optional[Path] = None,
    console: Optional[Console] = None,
    output_subdir: str = "prompt_entropy",
    client: Optional[AOAIClient] = None,
    executor_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, object]:
    """Execute prompt requests across worker shards with resumable progress tracking.

    The runner fans prompts out across `workers`, persists in-flight state to
    per-worker JSONL shards, and rebuilds the exact resume position if the
    process is stopped and restarted. Failed prompts are retried automatically
    on the next invocation unless their previous result succeeded.
    """
    if not requests:
        raise ValueError("run() requires at least one PromptRequest")

    if workers < 1:
        raise ValueError("workers must be >= 1")

    console = console or Console()
    output_dir = _ensure_output_dir(base_output_dir, subdir=output_subdir)
    console.print(f"[cyan]Output directory:[/] {output_dir}")

    if client is None:
        azure_cfg = AzureConfig.from_env()
        client = AOAIClient(azure_cfg)

    executor_kwargs = dict(executor_kwargs or {})
    allowed_executor_keys = {"max_concurrency", "on_start", "on_complete", "on_error", "on_retry"}
    unknown_executor_keys = set(executor_kwargs) - allowed_executor_keys
    if unknown_executor_keys:
        unknown_fmt = ", ".join(sorted(unknown_executor_keys))
        raise TypeError(f"Unsupported executor_kwargs for PromptRunner: {unknown_fmt}")

    max_concurrency_override = executor_kwargs.pop("max_concurrency", None)
    if max_concurrency_override not in (None, 1):
        raise ValueError(
            "PromptRunner requires max_concurrency=1 per worker to maintain deterministic resumes; "
            "set the `workers` argument instead."
        )

    executor_hooks = ExecutorHooks(
        on_start=executor_kwargs.pop("on_start", None),
        on_complete=executor_kwargs.pop("on_complete", None),
        on_error=executor_kwargs.pop("on_error", None),
        on_retry=executor_kwargs.pop("on_retry", None),
    )

    total_requests = len(requests)
    state = DashboardState(total=total_requests)

    worker_output_dir = output_dir / "worker_outputs"
    worker_output_dir.mkdir(parents=True, exist_ok=True)

    enumerated_requests = list(enumerate(requests))
    index_to_request = {idx: req for idx, req in enumerated_requests}
    # Round-robin split guarantees deterministic ownership per worker, even across resumes.
    assignments: List[List[Tuple[int, PromptRequest]]] = [
        [(idx, req) for idx, req in enumerated_requests if idx % workers == worker_id]
        for worker_id in range(workers)
    ]

    worker_paths: Dict[str, Path] = {
        f"worker_{worker_id:02d}": worker_output_dir / f"worker_{worker_id:02d}.jsonl"
        for worker_id in range(workers)
    }

    assigned_indices_per_worker: List[List[int]] = [
        [idx for idx, _ in pairs] for pairs in assignments
    ]
    worker_names: List[str] = [f"worker_{worker_id:02d}" for worker_id in range(workers)]
    worker_file_map: Dict[str, Path] = {
        path.stem: path for path in worker_output_dir.glob("worker_*.jsonl")
    }

    # Only carry forward successful prompts; failed prompts will be retried.
    existing_results_by_index: Dict[int, PromptResult] = {}
    existing_retries_by_index: Dict[int, int] = {}
    previously_failed_indices: set[int] = set()

    # Collect completed prompt records indexed by their global position so we can
    # safely resume even if the worker count changes between runs.
    for worker_name, path in sorted(worker_file_map.items()):
        if not path.exists():
            continue

        assigned_indices: List[int] = []
        try:
            worker_id = int(worker_name.split("_")[1])
        except (IndexError, ValueError):
            worker_id = -1
        if 0 <= worker_id < len(assigned_indices_per_worker):
            assigned_indices = assigned_indices_per_worker[worker_id]

        with path.open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                global_idx = data.get("index")
                if global_idx is None and line_idx < len(assigned_indices):
                    global_idx = assigned_indices[line_idx]

                if not isinstance(global_idx, int):
                    continue

                request = index_to_request.get(global_idx)
                if request is None:
                    continue

                result = _record_to_prompt_result(data, request, global_idx)
                if result.succeeded:
                    previously_failed_indices.discard(global_idx)
                    existing_results_by_index[global_idx] = result
                    existing_retries_by_index[global_idx] = max(result.total_calls - 1, 0)
                else:
                    previously_failed_indices.add(global_idx)

    existing_results = sorted(existing_results_by_index.values(), key=lambda r: r.index)
    existing_retries = sum(existing_retries_by_index.values())
    completed_indices = set(existing_results_by_index.keys())

    pending_assignments: List[List[Tuple[int, PromptRequest]]] = []
    pending_prefix_counts: List[List[int]] = []
    completed_counts_per_worker: List[int] = []
    for worker_id, assigned_pairs in enumerate(assignments):
        worker_pending: List[Tuple[int, PromptRequest]] = []
        worker_prefixes: List[int] = []
        completed_before = 0
        for global_idx, request in assigned_pairs:
            prefix_before = completed_before
            if global_idx in completed_indices:
                completed_before += 1
                continue
            worker_pending.append((global_idx, request))
            worker_prefixes.append(prefix_before)
        pending_assignments.append(worker_pending)
        pending_prefix_counts.append(worker_prefixes)
        completed_counts_per_worker.append(completed_before)

        total_assigned = len(assigned_pairs)
        pending_count = len(worker_pending)
        worker_name = worker_names[worker_id]
        has_retries_pending = any(idx in previously_failed_indices for idx, _ in worker_pending)
        if total_assigned == 0:
            status = "Idle"
        elif pending_count == 0:
            status = "Resume complete"
        elif has_retries_pending:
            status = f"Retry pending {pending_count}/{total_assigned}"
        elif completed_before > 0:
            status = f"Resume {completed_before}/{total_assigned}"
        else:
            status = f"Pending {pending_count}/{total_assigned}"
        state.worker_status[worker_name] = status

    # Count how many pending prompts correspond to prior failures so we can surface it.
    retry_target_count = sum(
        1
        for worker_pending in pending_assignments
        for idx, _ in worker_pending
        if idx in previously_failed_indices
    )

    if retry_target_count:
        plural_suffix = "s" if retry_target_count != 1 else ""
        console.print(
            f"[yellow]Retrying {retry_target_count} previously failed prompt{plural_suffix}.[/]"
        )

    state.started = len(existing_results)
    state.completed = sum(1 for r in existing_results if r.succeeded)
    state.failed = state.started - state.completed
    state.retries = existing_retries

    meta_path = output_dir / "run_meta.json"
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
    else:
        meta = {"resumes": 0}

    resumed_now = any(count > 0 for count in completed_counts_per_worker)
    if resumed_now:
        console.print("[yellow]Resuming previous run...[/]")
        meta["resumes"] = int(meta.get("resumes", 0)) + 1
    state.resumes = int(meta.get("resumes", 0))
    # Persist resume count so subsequent invocations keep accumulating the tally.
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f)

    run_start = time.time()
    new_results = _run_executor(
        client,
        assignments,
        pending_assignments,
        pending_prefix_counts,
        state,
        console,
        worker_paths,
        executor_hooks,
    )
    run_end = time.time()

    all_results = existing_results + new_results
    if not all_results:
        raise RuntimeError("PromptExecutor returned no results.")

    all_results.sort(key=lambda r: r.index)

    responses_path = output_dir / "responses.jsonl"
    stats_path = output_dir / "stats.yaml"

    all_records: List[Dict[str, Any]] = []
    pairs_by_worker_name: Dict[str, List[Tuple[int, PromptRequest]]] = {
        f"worker_{worker_id:02d}": assignments[worker_id] for worker_id in range(workers)
    }
    valid_indices = set(index_to_request.keys())
    for path in sorted(worker_output_dir.glob("worker_*.jsonl")):
        worker_name = path.stem
        pairs = pairs_by_worker_name.get(worker_name, [])
        with path.open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "index" not in record and line_idx < len(pairs):
                    record["index"] = pairs[line_idx][0]
                idx_val = record.get("index")
                if isinstance(idx_val, int) and idx_val not in valid_indices:
                    continue
                all_records.append(record)
        path.unlink(missing_ok=True)

    try:
        worker_output_dir.rmdir()
    except OSError:
        pass

    indexed_records: Dict[int, Dict[str, Any]] = {}
    unindexed_records: List[Dict[str, Any]] = []
    for record in all_records:
        idx_val = record.get("index")
        if isinstance(idx_val, int):
            indexed_records[idx_val] = record
        else:
            unindexed_records.append(record)

    ordered_records = [indexed_records[idx] for idx in sorted(indexed_records)] + unindexed_records
    with responses_path.open("w", encoding="utf-8") as out:
        for record in ordered_records:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    _write_stats(
        stats_path,
        all_results,
        run_started_at=run_start,
        run_ended_at=run_end,
        resumes=state.resumes,
    )

    successes = sum(1 for r in all_results if r.succeeded)
    console.print(
        f"[green]Run complete:[/] {successes}/{len(all_results)} succeeded. "
        f"Responses saved to [bold]{responses_path}[/], stats to [bold]{stats_path}[/]."
    )
    if state.resumes:
        console.print(f"[cyan]Total resumes recorded:[/] {state.resumes}")

    return {
        "results": all_results,
        "output_dir": output_dir,
        "responses_path": responses_path,
        "stats_path": stats_path,
        "resumes": state.resumes,
    }
