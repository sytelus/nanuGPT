#!/usr/bin/env python3
"""Generate descriptive analytics and visualisations for GSM8K graph runs.

This script mirrors the output-directory resolution used by
``gsm8k_graphs.py``: if ``--out_dir`` is not supplied, it falls back to
``$OUT_DIR`` or ``~/out_dir/gsm8k_graphs``. It expects the Arrow dataset
saved by ``gsm8k_graphs.py`` at ``<out_dir>/out_final/arrow_dataset``.

Outputs are written under ``<out_dir>/report`` and include a markdown
summary plus supporting plots and computational-graph diagrams for the
extreme examples of each measured quantity.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import textwrap
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch
except Exception as exc:  # pragma: no cover - guarded import
    raise SystemExit(
        "matplotlib is required for plotting. Install it before running this script."
    ) from exc

try:
    from datasets import load_from_disk
except Exception as exc:  # pragma: no cover - guarded import
    raise SystemExit(
        "datasets (🤗) library is required to read the Arrow dataset."
    ) from exc


@dataclass
class Quantity:
    key: str
    title: str
    description: str


@dataclass
class StatSummary:
    count: int
    mean: float
    median: float
    std: float
    minimum: float
    maximum: float


def resolve_output_dir(cli_out_dir: Optional[str]) -> str:
    """Match the directory resolution logic from gsm8k_graphs."""

    if cli_out_dir and str(cli_out_dir).strip():
        target = cli_out_dir
    else:
        base = os.environ.get("OUT_DIR", os.path.expanduser("~/out_dir"))
        target = os.path.join(base, "gsm8k_graphs")
    return os.path.abspath(target)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_arrow_dataset(out_dir: str):
    arrow_path = os.path.join(out_dir, "out_final", "arrow_dataset")
    if not os.path.exists(arrow_path):
        raise FileNotFoundError(
            f"Arrow dataset not found at {arrow_path}. Run gsm8k_graphs.py first."
        )
    return load_from_disk(arrow_path), arrow_path


def safe_float_list(values: Iterable[Any]) -> List[float]:
    result: List[float] = []
    for val in values:
        if val is None:
            continue
        try:
            result.append(float(val))
        except Exception:
            continue
    return result


def compute_statistics(values: Sequence[float]) -> Optional[StatSummary]:
    if not values:
        return None
    arr = np.array(values, dtype=float)
    return StatSummary(
        count=len(values),
        mean=float(np.mean(arr)),
        median=float(np.median(arr)),
        std=float(np.std(arr, ddof=0)),
        minimum=float(np.min(arr)),
        maximum=float(np.max(arr)),
    )


def histogram(values: Sequence[float], title: str, xlabel: str, path: str) -> None:
    if not values:
        return
    bins = min(50, max(10, int(math.sqrt(len(values)))))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=bins, color="#4a90e2", edgecolor="black", alpha=0.75)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def build_edges(nodes: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    edges: List[Tuple[str, str]] = []
    for node in nodes:
        if not isinstance(node, dict):
            continue
        nid = node.get("id")
        if not isinstance(nid, str):
            continue
        inputs = node.get("inputs") or []
        if isinstance(inputs, list):
            for src in inputs:
                if isinstance(src, str):
                    edges.append((src, nid))
    return edges


def topo_sort(nodes: List[Dict[str, Any]]) -> Tuple[List[str], Dict[str, int]]:
    id_to_node = {}
    indeg: Dict[str, int] = defaultdict(int)
    adj: Dict[str, List[str]] = defaultdict(list)

    for node in nodes:
        if not isinstance(node, dict):
            continue
        nid = node.get("id")
        if not isinstance(nid, str):
            continue
        id_to_node[nid] = node
        for src in node.get("inputs") or []:
            if isinstance(src, str):
                indeg[nid] += 1
                adj[src].append(nid)

    queue = [nid for nid in id_to_node if indeg[nid] == 0]
    order: List[str] = []
    level: Dict[str, int] = {nid: 0 for nid in id_to_node}

    from collections import deque

    dq = deque(queue)
    while dq:
        u = dq.popleft()
        order.append(u)
        for v in adj[u]:
            level[v] = max(level[v], level[u] + 1)
            indeg[v] -= 1
            if indeg[v] == 0:
                dq.append(v)

    if len(order) != len(id_to_node):
        # Graph may have cycles or references to missing nodes; best effort only.
        missing = set(id_to_node) - set(order)
        order.extend(missing)
        for nid in missing:
            level.setdefault(nid, 0)

    return order, level


def layout_positions(nodes: List[Dict[str, Any]]) -> Dict[str, Tuple[float, float]]:
    order, level = topo_sort(nodes)
    buckets: Dict[int, List[str]] = defaultdict(list)
    for nid in order:
        buckets[level[nid]].append(nid)

    positions: Dict[str, Tuple[float, float]] = {}
    for lvl in sorted(buckets):
        ids = buckets[lvl]
        width = max(len(ids) - 1, 1)
        for idx, nid in enumerate(ids):
            x = idx if len(ids) > 1 else 0.0
            x -= width / 2.0
            y = -float(lvl)
            positions[nid] = (x, y)
    return positions


def draw_graph_diagram(graph: Dict[str, Any], path: str, title: str) -> bool:
    nodes = graph.get("nodes") or []
    if not isinstance(nodes, list) or not nodes:
        return False

    edges = build_edges(nodes)
    positions = layout_positions(nodes)

    fig_w = max(6.0, len(set(level for _, level in positions.items())) * 2.0)
    fig_h = max(5.0, len(nodes) * 0.4)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Draw edges first so nodes overlay them.
    for src, dst in edges:
        if src not in positions or dst not in positions:
            continue
        sx, sy = positions[src]
        dx, dy = positions[dst]
        arrow = FancyArrowPatch(
            (sx, sy),
            (dx, dy),
            arrowstyle="->",
            color="#888888",
            linewidth=1.5,
            mutation_scale=12,
            alpha=0.9,
        )
        ax.add_patch(arrow)

    # Draw nodes with labels.
    for nid, (x, y) in positions.items():
        node = next((n for n in nodes if n.get("id") == nid), None)
        op = node.get("op") if isinstance(node, dict) else "?"
        display = f"{nid}\n{op}"
        ax.scatter([x], [y], s=300, color="#4a90e2", edgecolor="black", linewidth=1.0)
        ax.text(x, y, display, ha="center", va="center", fontsize=9, color="white")

    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


def format_float(value: Optional[float]) -> str:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "–"
    if abs(value) >= 1000 or (0 < abs(value) < 0.01):
        return f"{value:.3e}"
    return f"{value:.3f}"


def select_extremes(
    records: Sequence[Dict[str, Any]],
    key: str,
    count: int = 3,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    filtered = [r for r in records if r.get(key) is not None]
    if not filtered:
        return [], []
    filtered.sort(key=lambda r: r.get(key))
    lows = filtered[:count]
    highs = list(reversed(filtered[-count:]))
    return lows, highs


def summarise_boolean(values: Iterable[bool]) -> Tuple[int, int, float]:
    total = 0
    positives = 0
    for val in values:
        if val is None:
            continue
        total += 1
        positives += 1 if bool(val) else 0
    rate = (positives / total) if total > 0 else 0.0
    return positives, total, rate


def render_markdown(
    report_path: str,
    arrow_path: str,
    total_records: int,
    summary_lines: List[str],
    stat_rows: List[Tuple[str, Optional[StatSummary]]],
    quantity_sections: List[str],
) -> None:
    lines: List[str] = []
    lines.append("# GSM8K Graph Analysis Report")
    lines.append("")
    lines.append(f"*Dataset source*: `{arrow_path}`")
    lines.append(f"*Total problems analysed*: **{total_records}**")
    lines.append("")

    if summary_lines:
        lines.append("## Summary")
        lines.extend(summary_lines)
        lines.append("")

    lines.append("## Descriptive Statistics")
    lines.append("")
    header = "| Quantity | Count | Mean | Median | Std | Min | Max |"
    separator = "|---|---:|---:|---:|---:|---:|---:|"
    lines.append(header)
    lines.append(separator)
    for title, stats in stat_rows:
        if stats is None:
            row = f"| {title} | 0 | – | – | – | – | – |"
        else:
            row = "| {title} | {count} | {mean} | {median} | {std} | {minimum} | {maximum} |".format(
                title=title,
                count=stats.count,
                mean=format_float(stats.mean),
                median=format_float(stats.median),
                std=format_float(stats.std),
                minimum=format_float(stats.minimum),
                maximum=format_float(stats.maximum),
            )
        lines.append(row)
    lines.append("")

    lines.extend(quantity_sections)

    ensure_dir(os.path.dirname(report_path))
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Analyse GSM8K computational graph dataset.")
    parser.add_argument("--out_dir", "--outdir", dest="out_dir", type=str, default=None,
                        help="Base output directory (defaults to gsm8k_graphs resolve logic).")
    parser.add_argument("--report-subdir", type=str, default="report",
                        help="Name of the subdirectory to hold the report (default: report).")
    parser.add_argument("--extreme-samples", type=int, default=3,
                        help="Number of extreme examples to showcase per metric (default: 3).")
    args = parser.parse_args(argv)

    out_dir = resolve_output_dir(args.out_dir)
    report_dir = os.path.join(out_dir, args.report_subdir)
    images_dir = os.path.join(report_dir, "images")
    ensure_dir(images_dir)

    dataset, arrow_path = load_arrow_dataset(out_dir)
    records: List[Dict[str, Any]] = [dict(row) for row in dataset]

    # Augment derived columns needed for analysis.
    for rec in records:
        num_nodes = rec.get("num_nodes")
        num_edges = rec.get("num_edges")
        rec["sum_nodes_edges"] = (
            float(num_nodes) + float(num_edges)
            if num_nodes is not None and num_edges is not None
            else None
        )
        ops = rec.get("ops") or []
        rec["ops_count"] = len(ops) if isinstance(ops, list) else None

    total_records = len(records)

    quantities: List[Quantity] = [
        Quantity("num_nodes", "Number of Nodes", "Total nodes in the graph."),
        Quantity("num_edges", "Number of Edges", "Total directed edges implied by inputs."),
        Quantity("max_width", "Max Width", "Maximum nodes present at any depth level."),
        Quantity("height", "Height", "Number of graph levels (topological depth)."),
        Quantity("sum_nodes_edges", "Nodes + Edges", "Combined size of the graph."),
        Quantity("ops_count", "Unique Ops", "Distinct operator types used in the graph."),
    ]

    stat_rows: List[Tuple[str, Optional[StatSummary]]] = []
    quantity_sections: List[str] = []

    # Overall summary diagnostics.
    valid_count, valid_total, valid_rate = summarise_boolean(r.get("graph_valid") for r in records)
    eval_count, eval_total, eval_rate = summarise_boolean(r.get("eval_ok") for r in records)
    match_count, match_total, match_rate = summarise_boolean(r.get("answer_match") for r in records)

    summary_lines = [
        f"- Valid graphs: **{valid_count}/{valid_total}** ({valid_rate*100:.1f}%)",
        f"- Successfully evaluated: **{eval_count}/{eval_total}** ({eval_rate*100:.1f}%)",
        f"- Gold answer matches: **{match_count}/{match_total}** ({match_rate*100:.1f}%)",
        f"- Cached Arrow rows read: **{total_records}**",
    ]

    for qty in quantities:
        values = safe_float_list(rec.get(qty.key) for rec in records)
        stats = compute_statistics(values)
        stat_rows.append((qty.title, stats))

        section_lines: List[str] = []
        section_lines.append(f"## {qty.title}")
        section_lines.append("")
        section_lines.append(qty.description)
        section_lines.append("")

        if values:
            hist_fname = f"{qty.key}_hist.png"
            hist_path = os.path.join(images_dir, hist_fname)
            histogram(values, title=f"Distribution of {qty.title}", xlabel=qty.title, path=hist_path)
            section_lines.append(f"![Distribution of {qty.title}](images/{hist_fname})")
            section_lines.append("")
        else:
            section_lines.append("No data available for this quantity.")
            section_lines.append("")

        lows, highs = select_extremes(records, qty.key, count=args.extreme_samples)

        def describe_examples(label: str, examples: List[Dict[str, Any]]) -> None:
            section_lines.append(f"**{label} examples**")
            if not examples:
                section_lines.append("")
                section_lines.append("_No examples available._")
                section_lines.append("")
                return
            section_lines.append("")
            for rec in examples:
                value = rec.get(qty.key)
                rid = rec.get("id") or "?"
                question = rec.get("question") or "(question unavailable)"
                question_preview = textwrap.shorten(question.replace("\n", " "), width=160)
                graph_json = rec.get("graph_json")
                image_rel = None
                if graph_json:
                    try:
                        graph = json.loads(graph_json)
                        image_name = f"{qty.key}_{label.lower()}_{rid}.png".replace("/", "_")
                        image_path = os.path.join(images_dir, image_name)
                        ok = draw_graph_diagram(
                            graph,
                            image_path,
                            title=f"{qty.title} {label} example ({rid})",
                        )
                        if ok:
                            image_rel = f"images/{image_name}"
                    except Exception:
                        image_rel = None

                section_lines.append(f"- **{rid}** — {qty.title.lower()} = {format_float(value)}")
                section_lines.append(f"  - Question: {question_preview}")
                if image_rel:
                    section_lines.append(f"  - ![Graph for {rid}]({image_rel})")
                else:
                    section_lines.append("  - Graph diagram unavailable.")
            section_lines.append("")

        describe_examples("Minimum", lows)
        describe_examples("Maximum", highs)

        quantity_sections.extend(section_lines)

    report_path = os.path.join(report_dir, "report.md")
    render_markdown(
        report_path=report_path,
        arrow_path=arrow_path,
        total_records=total_records,
        summary_lines=summary_lines,
        stat_rows=stat_rows,
        quantity_sections=quantity_sections,
    )

    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
