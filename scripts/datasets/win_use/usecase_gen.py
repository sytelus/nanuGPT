from __future__ import annotations

import argparse
import json
import os
import random
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from nanugpt.api_client.aoai_client import AOAIClient, AzureConfig, ChatResult
from nanugpt.api_client.prompt_exec import PromptExecutor, PromptRequest, PromptResult


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


def build_dashboard(snapshot: Dict[str, object]) -> Panel:
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

    return Panel(
        Group(summary, worker_table),
        title="Prompt Execution Dashboard",
        border_style="magenta",
    )


def ensure_output_dir() -> Path:
    out_env = os.environ.get("OUT_DIR")
    base = Path(out_env) if out_env else Path.cwd()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    target = base / "prompt_entropy" / timestamp
    target.mkdir(parents=True, exist_ok=True)
    return target


def build_requests(total: int, rng: random.Random) -> List[PromptRequest]:
    numbers = list(range(1, total + 1))
    rng.shuffle(numbers)
    requests: List[PromptRequest] = []
    for idx, rnd in enumerate(numbers, start=1):
        prompt = PromptRequest(
            system_prompt="You are a witty assistant.",
            user_prompt=f"Tell me a joke {rnd}",
            metadata={"id": idx, "rnd": rnd},
        )
        requests.append(prompt)
    return requests


def write_jsonl(path: Path, results: List[PromptResult]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for res in results:
            metadata = res.request.metadata
            record = {
                "id": metadata.get("id"),
                "rnd": metadata.get("rnd"),
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


def write_stats(path: Path, results: List[PromptResult], start_ts: float, end_ts: float) -> None:
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
        "run_started_at": datetime.fromtimestamp(start_ts, tz=timezone.utc).isoformat(),
        "run_ended_at": datetime.fromtimestamp(end_ts, tz=timezone.utc).isoformat(),
        "run_elapsed_seconds": end_ts - start_ts,
        "average_api_time_seconds": (
            sum(r.api_duration for r in results) / len(results) if results else 0.0
        ),
        "average_total_calls": (
            sum(r.total_calls for r in results) / len(results) if results else 0.0
        ),
    }
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(stats, f, sort_keys=False)


def run_executor(
    executor: PromptExecutor,
    requests: List[PromptRequest],
    workers: int,
    state: DashboardState,
) -> List[PromptResult]:
    results_holder: Dict[str, List[PromptResult]] = {}
    error_holder: Dict[str, BaseException] = {}
    done_event = threading.Event()

    def make_status(metadata: Dict[str, object], prefix: str) -> str:
        prompt_id = metadata.get("id")
        rnd = metadata.get("rnd")
        return f"{prefix} id={prompt_id} rnd={rnd}"

    def on_start(index: int, request: PromptRequest, _result: PromptResult) -> None:
        worker = threading.current_thread().name
        with state.lock:
            state.started += 1
            state.worker_status[worker] = make_status(request.metadata, "Running")

    def on_complete(index: int, request: PromptRequest, chat_result: ChatResult, result: PromptResult) -> None:
        worker = threading.current_thread().name
        with state.lock:
            state.completed += 1
            state.worker_status[worker] = make_status(request.metadata, "Done")

    def on_error(index: int, request: PromptRequest, exc: BaseException, result: PromptResult) -> None:
        worker = threading.current_thread().name
        with state.lock:
            state.failed += 1
            state.worker_status[worker] = make_status(request.metadata, f"Error ({exc})")

    def on_retry(index: int, request: PromptRequest, attempt: int, exc: Exception) -> None:
        worker = threading.current_thread().name
        with state.lock:
            state.retries += 1
            state.worker_status[worker] = make_status(request.metadata, f"Retry #{attempt}")

    def target() -> None:
        try:
            results = executor.run_prompts(
                requests,
                concurrency=workers,
                on_start=on_start,
                on_complete=on_complete,
                on_error=on_error,
                on_retry=on_retry,
            )
            results_holder["results"] = results
        except BaseException as exc:  # capture any exception for re-raising
            error_holder["error"] = exc
        finally:
            done_event.set()

    thread = threading.Thread(target=target, name="PromptExecutorSupervisor", daemon=True)
    thread.start()

    console = Console()
    with Live(build_dashboard(state.snapshot()), console=console, refresh_per_second=5) as live:
        while not done_event.is_set():
            live.update(build_dashboard(state.snapshot()))
            time.sleep(0.2)
        # One final refresh after completion
        live.update(build_dashboard(state.snapshot()))

    thread.join()

    if "error" in error_holder:
        raise error_holder["error"]

    return results_holder.get("results", [])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate win_use prompts via Azure OpenAI.")
    parser.add_argument("--total", type=int, default=1000, help="Total number of prompts to run.")
    parser.add_argument("--workers", type=int, default=8, help="Maximum worker threads.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for shuffling prompts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    output_dir = ensure_output_dir()
    console = Console()
    console.print(f"[cyan]Output directory:[/] {output_dir}")

    try:
        azure_cfg = AzureConfig.from_env()
    except RuntimeError as exc:
        console.print(f"[red]Azure configuration error:[/] {exc}")
        raise SystemExit(1)

    client = AOAIClient(azure_cfg)
    executor = PromptExecutor(client, max_concurrency=args.workers)

    requests = build_requests(args.total, rng)
    state = DashboardState(total=len(requests))

    run_start = time.time()
    try:
        results = run_executor(executor, requests, args.workers, state)
    except BaseException as exc:
        console.print(f"[red]Execution failed:[/] {exc}")
        raise
    run_end = time.time()

    if not results:
        console.print("[red]No results were produced.[/]")
        raise SystemExit(1)

    responses_path = output_dir / "responses.jsonl"
    stats_path = output_dir / "stats.yaml"
    write_jsonl(responses_path, results)
    write_stats(stats_path, results, run_start, run_end)

    successes = sum(1 for r in results if r.succeeded)
    console.print(
        f"[green]Run complete:[/] {successes}/{len(results)} succeeded. "
        f"Responses saved to [bold]{responses_path}[/], stats to [bold]{stats_path}[/]."
    )


if __name__ == "__main__":
    main()
