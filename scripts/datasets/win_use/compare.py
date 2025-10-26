"""Compare two unique_responses.txt files and summarize overlaps."""

from __future__ import annotations

import argparse
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from rich import box
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

DEFAULT_SAMPLE_COUNT = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two unique_responses.txt files emitted by analysis.py, "
            "summarize the overlaps, and write a markdown report."
        )
    )
    parser.add_argument("left", help="Path to the first unique_responses.txt file.")
    parser.add_argument("right", help="Path to the second unique_responses.txt file.")
    parser.add_argument(
        "-n",
        "--samples",
        type=int,
        default=DEFAULT_SAMPLE_COUNT,
        help="How many sample items to display for each category (default: 5).",
    )
    parser.add_argument(
        "--report",
        help=(
            "Optional explicit report path. By default the script will create "
            "compare_<relative_left>_<relative_right>.md in the folder of the left file."
        ),
    )
    return parser.parse_args()


def load_unique_items(path: Path) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if not stripped or stripped in seen:
                continue
            seen.add(stripped)
            ordered.append(stripped)
    return ordered


def ordered_intersection(source: Iterable[str], target: set[str]) -> list[str]:
    return [item for item in source if item in target]


def ordered_difference(source: Iterable[str], other: set[str]) -> list[str]:
    return [item for item in source if item not in other]


def sample_items(items: list[str], limit: int) -> list[str]:
    if limit <= 0:
        return list(items)
    return items[:limit]


def relative_label(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def sanitize_fragment(value: str) -> str:
    interim = value.replace(os.sep, "_").replace("/", "_").replace("\\", "_")
    sanitized = re.sub(r"[^A-Za-z0-9._-]", "_", interim).strip("_")
    return sanitized or "file"


def default_report_path(left_path: Path, left_rel: str, right_rel: str) -> Path:
    fragment_left = sanitize_fragment(left_rel)
    fragment_right = sanitize_fragment(right_rel)
    filename = f"compare_{fragment_left}_{fragment_right}.md"
    return left_path.parent / filename


def ensure_file_exists(path: Path, label: str, console: Console) -> None:
    if not path.exists():
        console.print(f"[red]{label} file not found:[/] {path}")
        raise SystemExit(1)
    if not path.is_file():
        console.print(f"[red]{label} path is not a file:[/] {path}")
        raise SystemExit(1)


def render_console(
    console: Console,
    left_rel: str,
    right_rel: str,
    left_count: int,
    right_count: int,
    common: list[str],
    left_only: list[str],
    right_only: list[str],
    sample_limit: int,
) -> None:
    summary = Table(title="Unique Response Comparison", box=box.SIMPLE_HEAVY)
    summary.add_column("Metric", style="cyan", justify="left")
    summary.add_column("Value", style="magenta", justify="left")
    summary.add_row("Left file", escape(left_rel))
    summary.add_row("Right file", escape(right_rel))
    summary.add_row("Left items", str(left_count))
    summary.add_row("Right items", str(right_count))
    summary.add_row("Common items", str(len(common)))
    summary.add_row("Left-only items", str(len(left_only)))
    summary.add_row("Right-only items", str(len(right_only)))
    summary.add_row("Sample size", str(sample_limit if sample_limit > 0 else "all"))
    console.print(summary)

    panels = [
        Panel(
            _format_panel_body(common, sample_limit),
            title="Common Items",
            border_style="green",
        ),
        Panel(
            _format_panel_body(left_only, sample_limit),
            title="Left Only",
            border_style="yellow",
        ),
        Panel(
            _format_panel_body(right_only, sample_limit),
            title="Right Only",
            border_style="cyan",
        ),
    ]
    console.print(*panels, sep="\n\n")


def _format_panel_body(items: list[str], sample_limit: int) -> str:
    samples = sample_items(items, sample_limit)
    if not samples:
        return "[dim](no items)[/]"
    return "\n".join(escape(item) for item in samples)


def build_markdown(
    left_rel: str,
    right_rel: str,
    left_count: int,
    right_count: int,
    common: list[str],
    left_only: list[str],
    right_only: list[str],
    sample_limit: int,
) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    lines: list[str] = [
        "# Unique Response Comparison",
        "",
        f"- Generated: {timestamp}",
        f"- Left file: `{left_rel}` ({left_count} items)",
        f"- Right file: `{right_rel}` ({right_count} items)",
        f"- Common items: {len(common)}",
        f"- Left-only items: {len(left_only)}",
        f"- Right-only items: {len(right_only)}",
        f"- Sample size per section: {sample_limit if sample_limit > 0 else 'all'}",
        "",
    ]
    sections = [
        ("Common Items", common),
        ("Left Only", left_only),
        ("Right Only", right_only),
    ]
    for title, items in sections:
        lines.append(f"## {title}")
        lines.append("")
        lines.append(f"Count: {len(items)}")
        lines.append("")
        samples = sample_items(items, sample_limit)
        if samples:
            lines.append("```")
            lines.extend(samples)
            lines.append("```")
        else:
            lines.append("(no items)")
        lines.append("")
    return "\n".join(lines)


def write_report(report_path: Path, content: str) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    console = Console()

    left_path = Path(args.left).expanduser().resolve()
    right_path = Path(args.right).expanduser().resolve()

    ensure_file_exists(left_path, "Left", console)
    ensure_file_exists(right_path, "Right", console)

    left_items = load_unique_items(left_path)
    right_items = load_unique_items(right_path)

    left_set = set(left_items)
    right_set = set(right_items)

    common_items = ordered_intersection(left_items, right_set)
    left_only_items = ordered_difference(left_items, right_set)
    right_only_items = ordered_difference(right_items, left_set)

    sample_limit = args.samples
    base = Path.cwd()
    left_rel = relative_label(left_path, base)
    right_rel = relative_label(right_path, base)

    report_path = (
        Path(args.report).expanduser().resolve()
        if args.report
        else default_report_path(left_path, left_rel, right_rel)
    )

    render_console(
        console,
        left_rel,
        right_rel,
        len(left_items),
        len(right_items),
        common_items,
        left_only_items,
        right_only_items,
        sample_limit,
    )

    report_content = build_markdown(
        left_rel,
        right_rel,
        len(left_items),
        len(right_items),
        common_items,
        left_only_items,
        right_only_items,
        sample_limit,
    )
    write_report(report_path, report_content)
    console.print(f"[bold]Report:[/] {report_path}")


if __name__ == "__main__":
    main()

