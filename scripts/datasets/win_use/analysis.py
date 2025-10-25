"""Analyze prompt entropy outputs, summarize duplicates, and write a markdown report."""

from __future__ import annotations

import argparse
import json
import os
import statistics
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from rich import box
from rich.console import Console
from rich.table import Table

OUT_DIR = Path(os.environ.get("OUT_DIR", Path.cwd()))
DEFAULT_SUBDIR = "prompt_entropy_exec_phrases_i5"
DEFAULT_INPUT = OUT_DIR / DEFAULT_SUBDIR / "responses.jsonl"
DEFAULT_REPORT = OUT_DIR / DEFAULT_SUBDIR / "report.md"
DEFAULT_UNIQUE_RESPONSES = OUT_DIR / DEFAULT_SUBDIR / "unique_responses.txt"
ITEMS_PER_RESPONSE = 1


@dataclass(frozen=True)
class LengthStats:
    mean: float
    median: float
    shortest: int
    longest: int


@dataclass(frozen=True)
class AnalysisResult:
    total_records: int
    error_records: int
    malformed_records: int
    missing_response_records: int
    blank_response_records: int
    counts: Counter[str]
    lengths: LengthStats

    @property
    def valid_count(self) -> int:
        return sum(self.counts.values())

    @property
    def unique_count(self) -> int:
        return len(self.counts)

    @property
    def duplicate_count(self) -> int:
        return self.valid_count - self.unique_count

    @property
    def unique_ratio(self) -> float:
        return (self.unique_count / self.valid_count) if self.valid_count else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze prompt entropy output JSONL, summarize duplicates, "
            "and emit a markdown report."
        )
    )
    parser.add_argument(
        "-i",
        "--input",
        default=str(DEFAULT_INPUT),
        help="Path to prompt entropy responses JSONL (default: prompt_entropy_exec_phrases_i/responses.jsonl).",
    )
    parser.add_argument(
        "-o",
        "--report",
        default=str(DEFAULT_REPORT),
        help="Path to write markdown report (default: scripts/datasets/win_use/report.md).",
    )
    parser.add_argument(
        "-t",
        "--top",
        type=int,
        default=20,
        help="How many rows of the frequency table to show in the terminal (default: 20).",
    )
    parser.add_argument(
        "-u",
        "--unique",
        default=str(DEFAULT_UNIQUE_RESPONSES),
        help="Path to write newline-delimited unique responses (default: unique_responses.txt).",
    )
    return parser.parse_args()


def load_records(path: Path) -> AnalysisResult:
    total = error = malformed = missing = blank = 0
    counts: Counter[str] = Counter()
    lengths: list[int] = []

    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, 1):
            stripped = raw.strip()
            if not stripped:
                continue
            total += 1
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                malformed += 1
                continue

            error_val = record.get("error")
            if error_val not in (None, "", 0):
                error += 1
                continue

            response = record.get("response")
            if response is None:
                missing += 1
                continue

            text = str(response).strip()
            if not text:
                blank += 1
            counts[text] += 1
            lengths.append(len(text))

    if lengths:
        length_stats = LengthStats(
            mean=statistics.mean(lengths),
            median=statistics.median(lengths),
            shortest=min(lengths),
            longest=max(lengths),
        )
    else:
        length_stats = LengthStats(mean=0.0, median=0.0, shortest=0, longest=0)

    return AnalysisResult(
        total_records=total,
        error_records=error,
        malformed_records=malformed,
        missing_response_records=missing,
        blank_response_records=blank,
        counts=counts,
        lengths=length_stats,
    )


def md_escape(value: str) -> str:
    escaped = value.replace("|", r"\|").replace("\n", " ")
    return escaped if escaped else "(empty response)"


def summarize_response(response: str, items_per_response: int = ITEMS_PER_RESPONSE) -> str:
    if not response:
        return ""
    lines = [line.strip() for line in response.splitlines()]
    if not lines:
        return ""
    return "\n".join(lines[:items_per_response])


def response_item_label(items_per_response: int = ITEMS_PER_RESPONSE) -> str:
    if items_per_response == 1:
        return "First line"
    return f"First {items_per_response} lines"


def make_markdown(result: AnalysisResult, input_path: Path) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    items_label = response_item_label()
    lines = [
        "# prompt entropy Response Analysis",
        "",
        f"- Generated: {timestamp}",
        f"- Input file: `{input_path}`",
        f"- Total records: {result.total_records}",
        f"- Records with error: {result.error_records}",
        f"- Records with malformed JSON: {result.malformed_records}",
        f"- Records missing response: {result.missing_response_records}",
        f"- Valid responses: {result.valid_count}",
        f"- Unique responses: {result.unique_count}",
        f"- Unique/valid ratio: {result.unique_ratio:.4f}",
        f"- Duplicate responses: {result.duplicate_count}",
        f"- Blank (empty) responses: {result.blank_response_records}",
        f"- Avg chars per response: {result.lengths.mean:.1f}",
        f"- Median chars per response: {result.lengths.median:.1f}",
        f"- Shortest response length: {result.lengths.shortest}",
        f"- Longest response length: {result.lengths.longest}",
        "",
        "## Response Frequency",
        "",
        f"| Count | {items_label} |",
        "| ---: | --- |",
    ]
    for response, count in result.counts.most_common():
        summary = summarize_response(response)
        lines.append(f"| {count} | {md_escape(summary)} |")
    return "\n".join(lines) + "\n"


def render_console(console: Console, result: AnalysisResult, top_n: int) -> None:
    summary = Table(title="Exec Phrase Summary", box=box.SIMPLE_HEAVY)
    summary.add_column("Metric", justify="left", style="cyan")
    summary.add_column("Value", justify="right", style="bold")
    summary.add_row("Total records", f"{result.total_records}")
    summary.add_row("Errors", f"{result.error_records}")
    summary.add_row("Malformed", f"{result.malformed_records}")
    summary.add_row("Missing response", f"{result.missing_response_records}")
    summary.add_row("Valid responses", f"{result.valid_count}")
    summary.add_row("Unique responses", f"{result.unique_count}")
    summary.add_row("Unique ratio", f"{result.unique_ratio:.2%}")
    summary.add_row("Duplicates", f"{result.duplicate_count}")
    summary.add_row("Blank responses", f"{result.blank_response_records}")
    summary.add_row("Avg chars", f"{result.lengths.mean:.1f}")
    summary.add_row("Median chars", f"{result.lengths.median:.1f}")
    summary.add_row("Range", f"{result.lengths.shortest}–{result.lengths.longest}")
    console.print(summary)

    items_label = response_item_label()
    freq = Table(
        title=f"Top {min(top_n, result.unique_count)} Responses ({items_label.lower()})",
        box=box.SIMPLE,
    )
    freq.add_column("Count", justify="right", style="magenta")
    freq.add_column(items_label, style="green")

    display_rows = result.counts.most_common(top_n if top_n > 0 else result.unique_count)
    for response, count in display_rows:
        summary = summarize_response(response)
        freq.add_row(str(count), summary or "(empty response)")
    if result.unique_count > len(display_rows):
        freq.caption = f"... {result.unique_count - len(display_rows)} additional unique responses in report."
    console.print(freq)


def write_report(report_path: Path, content: str) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(content, encoding="utf-8")


def write_unique_responses(unique_path: Path, responses: Iterable[str]) -> None:
    unique_path.parent.mkdir(parents=True, exist_ok=True)
    with unique_path.open("w", encoding="utf-8") as fh:
        for response in responses:
            sanitized = response.replace("\r\n", "\n").replace("\r", "\n").replace("\n", " ")
            fh.write(sanitized)
            fh.write("\n")


def main() -> None:
    args = parse_args()
    console = Console()

    input_path = Path(args.input).expanduser()
    report_path = Path(args.report).expanduser()
    unique_path = Path(args.unique).expanduser()
    if not input_path.exists():
        console.print(f"[red]Input file not found:[/] {input_path}")
        raise SystemExit(1)

    result = load_records(input_path)
    report_body = make_markdown(result, input_path)
    write_report(report_path, report_body)
    write_unique_responses(unique_path, result.counts.keys())

    console.print(f"[bold]Input file:[/] {input_path}")
    console.print(f"[bold]Report file:[/] {report_path}")
    console.print(f"[bold]Unique responses file:[/] {unique_path}")
    render_console(console, result, args.top)


if __name__ == "__main__":
    main()
