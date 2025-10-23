from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import yaml
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from .aoai_client import ChatResult
from .prompt_exec import PromptExecutor, PromptRequest, PromptResult


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
            }


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
    out_env = os.environ.get("OUT_DIR")
    base = base_output_dir or (Path(out_env) if out_env else Path.cwd())
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    target = base / subdir / timestamp
    target.mkdir(parents=True, exist_ok=True)
    return target


def _write_jsonl(path: Path, results: Sequence[PromptResult]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for res in results:
            metadata = res.request.metadata
            record = {
                "id": metadata.get("id"),
                "metadata": metadata,
                "status": "succeeded" if res.succeeded else "failed",
                "response": res.chat_result.content if res.chat_result else None,
                "error": str(res.error) if res.error else None,
                "api_duration": res.api_duration,
                "retry_duration": res.retry_duration,
                "total_calls": res.total_calls,
                "input_tokens": res.input_tokens,
                "output_tokens": res.output_tokens,
                "total_tokens": res.total_tokens,
                "started_at": res.started_at,
                "ended_at": res.ended_at,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_stats(
    path: Path,
    results: Sequence[PromptResult],
    *,
    run_started_at: float,
    run_ended_at: float,
) -> None:
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
    }
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(stats, f, sort_keys=False)


def _run_executor(
    executor: PromptExecutor,
    requests: Sequence[PromptRequest],
    workers: int,
    state: DashboardState,
    console: Console,
) -> List[PromptResult]:
    results_holder: Dict[str, List[PromptResult]] = {}
    done_event = threading.Event()

    def _format_status(prefix: str, metadata: Dict[str, object], extra: Optional[str] = None) -> str:
        parts = [prefix]
        if "id" in metadata:
            parts.append(f"id={metadata['id']}")
        if "rnd" in metadata:
            parts.append(f"rnd={metadata['rnd']}")
        if extra:
            parts.append(extra)
        return " ".join(parts)

    def on_start(index: int, request: PromptRequest, _result: PromptResult) -> None:
        worker = threading.current_thread().name
        with state.lock:
            state.started += 1
            state.worker_status[worker] = _format_status("Running", request.metadata)

    def on_complete(index: int, request: PromptRequest, chat_result: ChatResult, result: PromptResult) -> None:
        worker = threading.current_thread().name
        with state.lock:
            state.completed += 1
            state.worker_status[worker] = _format_status("Done", request.metadata)

    def on_error(index: int, request: PromptRequest, exc: BaseException, result: PromptResult) -> None:
        worker = threading.current_thread().name
        with state.lock:
            state.failed += 1
            state.worker_status[worker] = _format_status("Error", request.metadata, str(exc))

    def on_retry(index: int, request: PromptRequest, attempt: int, exc: Exception) -> None:
        worker = threading.current_thread().name
        with state.lock:
            state.retries += 1
            state.worker_status[worker] = _format_status("Retry", request.metadata, f"#{attempt}")

    def target() -> None:
        results = executor.run_prompts(
            list(requests),
            concurrency=workers,
            on_start=on_start,
            on_complete=on_complete,
            on_error=on_error,
            on_retry=on_retry,
        )
        results_holder["results"] = results
        done_event.set()

    thread = threading.Thread(target=target, name="PromptExecutorSupervisor", daemon=True)
    thread.start()

    with Live(_build_dashboard(state.snapshot()), console=console, refresh_per_second=5) as live:
        while not done_event.is_set():
            live.update(_build_dashboard(state.snapshot()))
            time.sleep(0.2)
        live.update(_build_dashboard(state.snapshot()))

    thread.join()

    return results_holder.get("results", [])


def run(
    executor: PromptExecutor,
    requests: Sequence[PromptRequest],
    *,
    workers: int,
    base_output_dir: Optional[Path] = None,
    console: Optional[Console] = None,
    output_subdir: str = "prompt_entropy",
) -> Dict[str, object]:
    if not requests:
        raise ValueError("run() requires at least one PromptRequest")

    console = console or Console()
    output_dir = _ensure_output_dir(base_output_dir, subdir=output_subdir)
    console.print(f"[cyan]Output directory:[/] {output_dir}")

    state = DashboardState(total=len(requests))

    run_start = time.time()
    results = _run_executor(executor, requests, workers, state, console)
    run_end = time.time()

    if not results:
        raise RuntimeError("PromptExecutor returned no results.")

    responses_path = output_dir / "responses.jsonl"
    stats_path = output_dir / "stats.yaml"
    _write_jsonl(responses_path, results)
    _write_stats(stats_path, results, run_started_at=run_start, run_ended_at=run_end)

    successes = sum(1 for r in results if r.succeeded)
    console.print(
        f"[green]Run complete:[/] {successes}/{len(results)} succeeded. "
        f"Responses saved to [bold]{responses_path}[/], stats to [bold]{stats_path}[/]."
    )

    return {
        "results": results,
        "output_dir": output_dir,
        "responses_path": responses_path,
        "stats_path": stats_path,
    }
