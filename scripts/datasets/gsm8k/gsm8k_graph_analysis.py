#!/usr/bin/env python3
"""Generate descriptive analytics and visualisations for GSM8K graph runs.

Purpose
-------
Transform the cached Arrow dataset produced by ``gsm8k_graphs.py`` into a
curated markdown report that highlights structural properties of the
generated computational graphs. The intent is to provide a quick, repeatable
overview of run quality and interesting edge cases without re-reading raw
JSON caches.

Inputs & Discovery
------------------
The script searches for the dataset using the same resolution rules as
``gsm8k_graphs.py``: an explicit ``--out_dir`` overrides environment
``$OUT_DIR`` (defaulting to ``~/out_dir/gsm8k_graphs``). It expects the Arrow
dataset directory at ``<out_dir>/out_final/arrow_dataset``.

Outputs
-------
Artifacts are written to ``<out_dir>/<report-subdir>`` (default ``report``):

* ``report.md`` – markdown narrative with summary stats, tables, and extreme
  examples per metric.
* ``images/*.png`` – histogram visualisations for each measured quantity and
  diagrammatic renderings of selected computational graphs.
* Additional tables covering operator usage and outcome-comparison insights.

Workflow Summary
----------------
1. Load the Arrow dataset into memory as plain dictionaries.
2. Enrich records with derived metrics (e.g., nodes+edges, width/height ratio,
   fan-in statistics, unique op count) and aggregate global operator usage.
3. Compute descriptive statistics and save distribution histograms.
4. Identify extreme examples (lowest/highest) per metric and render their
   graphs using a simple topological layout.
5. Compare matched vs. mismatched outcomes for each metric.
6. Assemble all findings into a markdown report with inline image references,
   tables, and commentary.

Graph Rendering Overview
------------------------
Nodes are arranged by topological depth; within each depth bucket they are
spaced evenly along the x-axis. Directed edges are drawn as arrow patches.
The renderer is defensive—invalid graphs are skipped rather than crash the
report generation pipeline.

Extensibility Notes
-------------------
* New metrics can be added by appending to the ``quantities`` list and
  providing any additional derived fields necessary in the preprocessing loop.
* The plotting utilities centralise styling so future tweaks (e.g., seaborn)
  only require changes in one place.
* This script deliberately avoids relying on private structures from
  ``gsm8k_graphs.py``; it only consumes data persisted in the Arrow dataset.
* Lightweight console logging is handled via the ``log`` helper; replace with a
  richer logger if structured output is required.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import textwrap
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, Circle
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
    interpretation: str


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


def log(message: str) -> None:
    """Lightweight progress logging to stdout."""
    print(f"[gsm8k_analysis] {message}")


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

    node_lookup = {
        node.get("id"): node
        for node in nodes
        if isinstance(node, dict) and isinstance(node.get("id"), str)
    }

    edges = build_edges(nodes)
    positions = layout_positions(nodes)
    if not positions:
        return False

    xs = [pos[0] for pos in positions.values()]
    ys = [pos[1] for pos in positions.values()]
    x_span = (max(xs) - min(xs)) if xs else 1.0
    y_span = (max(ys) - min(ys)) if ys else 1.0
    fig_w = max(6.0, x_span * 1.6 + 2.5)
    fig_h = max(4.0, y_span * 1.6 + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_aspect("equal", adjustable="datalim")

    node_radius = 0.22

    # Draw edges first so nodes overlay them.
    for src, dst in edges:
        if src not in positions or dst not in positions:
            continue
        sx, sy = positions[src]
        tx, ty = positions[dst]
        vec_x = tx - sx
        vec_y = ty - sy
        distance = math.hypot(vec_x, vec_y)
        if distance == 0:
            continue
        norm_x = vec_x / distance
        norm_y = vec_y / distance
        start = (sx + norm_x * (node_radius + 0.02), sy + norm_y * (node_radius + 0.02))
        end = (tx - norm_x * (node_radius + 0.04), ty - norm_y * (node_radius + 0.04))
        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            color="#6b6f72",
            linewidth=1.6,
            mutation_scale=15,
            alpha=0.9,
            zorder=1,
        )
        ax.add_patch(arrow)

    # Draw nodes with orbiting labels so longer text remains readable.
    for nid, (x, y) in positions.items():
        circle = Circle(
            (x, y),
            radius=node_radius,
            facecolor="#4a90e2",
            edgecolor="#1f4b75",
            linewidth=1.2,
            alpha=0.95,
            zorder=2,
        )
        ax.add_patch(circle)

        node = node_lookup.get(nid)
        op = node.get("op") if isinstance(node, dict) else None
        op_label = str(op) if op is not None else "?"
        display = f"{nid}\n{op_label}"
        label_lines = display.splitlines()
        max_chars = max((len(line) for line in label_lines), default=0)
        if max_chars <= 8:
            label_fontsize = 11
        elif max_chars <= 16:
            label_fontsize = 10
        elif max_chars <= 28:
            label_fontsize = 9
        else:
            label_fontsize = 8

        text_x = x + node_radius + 0.12
        ax.text(
            text_x,
            y,
            display,
            ha="left",
            va="center",
            fontsize=label_fontsize,
            color="#1b1b1b",
            linespacing=1.2,
            bbox={
                "boxstyle": "round,pad=0.25",
                "fc": "#ffffff",
                "ec": "#d0d4d8",
                "alpha": 0.95,
            },
            zorder=3,
        )

    margin_x = max(0.6, node_radius + 0.4)
    margin_y = max(0.6, node_radius + 0.4)
    if xs:
        ax.set_xlim(min(xs) - margin_x, max(xs) + margin_x)
    if ys:
        ax.set_ylim(min(ys) - margin_y, max(ys) + margin_y)

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


def github_slug(text: str) -> str:
    slug_chars: List[str] = []
    last_dash = False
    for ch in text.lower():
        if ch.isalnum():
            slug_chars.append(ch)
            last_dash = False
        elif ch in {" ", "-", "_"}:
            if not last_dash:
                slug_chars.append("-")
                last_dash = True
        else:
            continue
    slug = "".join(slug_chars).strip("-")
    return slug or "section"


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


def select_nearest(
    records: Sequence[Dict[str, Any]],
    key: str,
    target: Optional[float],
) -> Optional[Dict[str, Any]]:
    if target is None or math.isnan(target) or math.isinf(target):
        return None

    closest: Optional[Dict[str, Any]] = None
    best_delta = float("inf")
    for rec in records:
        value = rec.get(key)
        if value is None:
            continue
        try:
            as_float = float(value)
        except (TypeError, ValueError):
            continue
        delta = abs(as_float - target)
        if delta < best_delta:
            best_delta = delta
            closest = rec
    return closest


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

    lines.append("## Table of Contents")
    lines.append("")
    if summary_lines:
        lines.append("- [Summary](#summary)")
    lines.append("- [Descriptive Statistics](#descriptive-statistics)")
    lines.append("- [Metric Deep Dives](#metric-deep-dives)")
    if stat_rows:
        for title, _ in stat_rows:
            slug = github_slug(title)
            lines.append(f"  - [{title}](#{slug})")
    lines.append("- [Operator Landscape](#operator-landscape)")
    lines.append("- [Outcome Comparison](#outcome-comparison)")
    lines.append("")

    if summary_lines:
        lines.append("## Summary")
        lines.extend(summary_lines)
        lines.append("")

    lines.append("## Descriptive Statistics")
    lines.append("")
    lines.append(
        textwrap.fill(
            "Summary statistics capture the central tendencies of each quantity before the report "
            "dives into detailed commentary and examples.",
            width=100,
        )
    )
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

    log(f"Resolved output directory: {out_dir}")
    log(f"Report directory: {report_dir}")

    dataset, arrow_path = load_arrow_dataset(out_dir)
    log(f"Loading Arrow dataset from {arrow_path} ...")
    records: List[Dict[str, Any]] = [dict(row) for row in dataset]
    log(f"Loaded {len(records)} records. Deriving per-graph metrics...")

    # Global accumulators for cross-cutting analysis.
    ops_counter: Counter[str] = Counter()

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

        rec["edges_per_node"] = (
            (float(num_edges) / float(num_nodes))
            if num_nodes not in (None, 0) and num_edges is not None
            else None
        )
        height = rec.get("height")
        max_width = rec.get("max_width")
        rec["width_to_height"] = (
            (float(max_width) / float(height))
            if height not in (None, 0) and max_width is not None
            else None
        )

        # Parse level histogram to compute average width for memory style insights.
        level_counts: List[int] = []
        for entry in rec.get("levels") or []:
            if isinstance(entry, (int, float)):
                level_counts.append(int(entry))
                continue
            if isinstance(entry, str) and ":" in entry:
                _, count_str = entry.split(":", 1)
                try:
                    level_counts.append(int(count_str))
                except ValueError:
                    continue
        rec["avg_width"] = (
            float(sum(level_counts)) / float(len(level_counts))
            if level_counts
            else None
        )

        graph_json = rec.get("graph_json")
        graph_obj: Optional[Dict[str, Any]] = None
        if graph_json:
            try:
                graph_obj = json.loads(graph_json)
            except Exception:
                graph_obj = None

        if graph_obj:
            nodes = graph_obj.get("nodes") or []
            input_like = 0
            input_counts: List[int] = []
            for node in nodes:
                if not isinstance(node, dict):
                    continue
                op = node.get("op")
                if isinstance(op, str):
                    ops_counter[op] += 1
                    if op.lower() in {"input", "const"}:
                        input_like += 1
                ins = node.get("inputs") or []
                if isinstance(ins, list):
                    input_counts.append(len(ins))
                else:
                    input_counts.append(0)

            total_nodes = len(nodes)
            rec["avg_inputs_per_node"] = (
                float(sum(input_counts)) / float(total_nodes)
                if total_nodes > 0
                else None
            )
            rec["max_inputs_per_node"] = (
                float(max(input_counts))
                if input_counts
                else None
            )
            rec["input_const_ratio"] = (
                float(input_like) / float(total_nodes)
                if total_nodes > 0
                else None
            )
        else:
            rec["avg_inputs_per_node"] = None
            rec["max_inputs_per_node"] = None
            rec["input_const_ratio"] = None

    log("Derived metrics complete.")
    total_records = len(records)

    quantities: List[Quantity] = [
        Quantity(
            "num_nodes",
            "Number of Nodes",
            "Total node count in the generated graph.",
            "Acts as a proxy for overall computational footprint—larger graphs may require more execution steps and storage.",
        ),
        Quantity(
            "num_edges",
            "Number of Edges",
            "Total directed edges implied by node inputs.",
            "Edges signal data dependencies; more edges typically translate to heavier data movement and intermediate storage.",
        ),
        Quantity(
            "sum_nodes_edges",
            "Nodes + Edges",
            "Aggregate size combining nodes and edges.",
            "Useful for ranking heavy problems because it blends structural breadth and dependency count into a single magnitude.",
        ),
        Quantity(
            "max_width",
            "Max Width",
            "Peak number of nodes at any depth level.",
            "Correlates with instantaneous memory pressure—the wider the layer, the more parallel state must be held.",
        ),
        Quantity(
            "avg_width",
            "Average Width",
            "Mean number of nodes per depth level (derived from level histogram).",
            "Highlights how consistently wide the computation remains; high averages imply sustained parallel workloads.",
        ),
        Quantity(
            "height",
            "Height",
            "Number of depth levels in the graph (topological height).",
            "Captures sequential work: deeper graphs generally demand more serial reasoning steps.",
        ),
        Quantity(
            "width_to_height",
            "Width-to-Height Ratio",
            "Max width divided by height.",
            "A heuristic for parallelism vs. sequential depth—high ratios suggest wide, shallow computations.",
        ),
        Quantity(
            "edges_per_node",
            "Edges per Node",
            "Average outgoing dependencies per node (edges ÷ nodes).",
            "Approximates branching factor; higher values imply more fan-in/out to manage per operation.",
        ),
        Quantity(
            "avg_inputs_per_node",
            "Avg Inputs per Node",
            "Mean number of inputs provided to each node (parsed from graph JSON).",
            "Highlights combinational complexity—nodes with many inputs often encode larger algebraic steps.",
        ),
        Quantity(
            "max_inputs_per_node",
            "Max Inputs per Node",
            "Largest fan-in observed for any node in the graph.",
            "Isolates extreme aggregation points that may dominate compute or memory needs.",
        ),
        Quantity(
            "input_const_ratio",
            "Input/Const Ratio",
            "Fraction of nodes that are Input or Const (sourced from graph JSON).",
            "Measures how much of the graph is spent wiring known values versus performing transformations—lower ratios imply richer computation.",
        ),
        Quantity(
            "ops_count",
            "Unique Ops",
            "Number of distinct operator types used in the graph.",
            "Variety of operators hints at conceptual breadth; diverse graphs may exercise more reasoning templates.",
        ),
    ]

    stat_rows: List[Tuple[str, Optional[StatSummary]]] = []
    quantity_sections: List[str] = ["## Metric Deep Dives", ""]

    # Overall summary diagnostics.
    valid_count, valid_total, valid_rate = summarise_boolean(r.get("graph_valid") for r in records)
    eval_count, eval_total, eval_rate = summarise_boolean(r.get("eval_ok") for r in records)
    match_count, match_total, match_rate = summarise_boolean(r.get("answer_match") for r in records)

    summary_lines = [
        textwrap.fill(
            "High-level run health indicators provide context for the deeper analyses below.",
            width=100,
        ),
        "",
        f"- Valid graphs: **{valid_count}/{valid_total}** ({valid_rate*100:.1f}%)",
        f"- Successfully evaluated: **{eval_count}/{eval_total}** ({eval_rate*100:.1f}%)",
        f"- Gold answer matches: **{match_count}/{match_total}** ({match_rate*100:.1f}%)",
        f"- Cached Arrow rows read: **{total_records}**",
    ]

    for qty in quantities:
        log(f"Analyzing metric: {qty.title}")
        values = safe_float_list(rec.get(qty.key) for rec in records)
        stats = compute_statistics(values)
        stat_rows.append((qty.title, stats))

        section_lines: List[str] = []
        section_lines.append(f"### {qty.title}")
        section_lines.append("")
        section_lines.append(textwrap.fill(qty.description, width=100))
        section_lines.append("")
        section_lines.append(textwrap.fill(qty.interpretation, width=100))
        section_lines.append("")

        if values:
            hist_fname = f"{qty.key}_hist.png"
            hist_path = os.path.join(images_dir, hist_fname)
            log(f"  - Writing histogram to {hist_path}")
            histogram(values, title=f"Distribution of {qty.title}", xlabel=qty.title, path=hist_path)
            section_lines.append(f"![Distribution of {qty.title}](images/{hist_fname})")
            section_lines.append("")
        else:
            section_lines.append("No data available for this quantity.")
            section_lines.append("")

        lows, highs = select_extremes(records, qty.key, count=args.extreme_samples)

        def describe_examples(label: str, examples: List[Dict[str, Any]]) -> None:
            heading = "example" if len(examples) == 1 else "examples"
            section_lines.append(f"**{label} {heading}**")
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
                answer = rec.get("answer") or "(answer unavailable)"
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
                section_lines.append("  - Question:")
                section_lines.append("")
                section_lines.append("    ```text")
                question_text = question if isinstance(question, str) else str(question)
                question_lines = question_text.splitlines() or [question_text]
                for line in question_lines:
                    section_lines.append(f"    {line}")
                section_lines.append("    ```")
                section_lines.append("")
                section_lines.append("  - Answer:")
                section_lines.append("")
                section_lines.append("    ```text")
                answer_text = answer if isinstance(answer, str) else str(answer)
                answer_lines = answer_text.splitlines() or [answer_text]
                for line in answer_lines:
                    section_lines.append(f"    {line}")
                section_lines.append("    ```")
                section_lines.append("")
                if image_rel:
                    section_lines.append(f"  - ![Graph for {rid}]({image_rel})")
                else:
                    section_lines.append("  - Graph diagram unavailable.")
            section_lines.append("")

        central_label: Optional[str] = None
        central_example: Optional[Dict[str, Any]] = None
        if stats is not None:
            target_value = stats.median
            central_label = "Median"
            if math.isnan(target_value) or math.isinf(target_value):
                target_value = stats.mean
                central_label = "Mean"
            central_example = select_nearest(records, qty.key, target_value)
            if central_example is None and central_label == "Median":
                central_label = "Mean"
                central_example = select_nearest(records, qty.key, stats.mean)

        describe_examples("Minimum", lows)
        if central_example and central_label:
            describe_examples(central_label, [central_example])
        describe_examples("Maximum", highs)

        quantity_sections.extend(section_lines)

    # Operator usage overview
    log("Summarizing operator usage ...")
    operator_sections: List[str] = [
        "## Operator Landscape",
        "",
        textwrap.fill(
            "Operator frequencies offer a quick lens on the reasoning templates exercised by the run. High-usage "
            "operators may warrant focused optimisation or qualitative review.",
            width=100,
        ),
        "",
    ]

    if ops_counter:
        operator_sections.append("| Operator | Count | Share |")
        operator_sections.append("|---|---:|---:|")
        total_ops = sum(ops_counter.values()) or 1
        for op, count in ops_counter.most_common(20):
            share = count / total_ops
            operator_sections.append(
                f"| `{op}` | {count} | {share*100:.1f}% |"
            )
        operator_sections.append("")
    else:
        operator_sections.append("_No operator data available (no graphs parsed)._")
        operator_sections.append("")

    quantity_sections.extend(operator_sections)

    # Compare outcomes (answer match) across metrics
    log("Comparing metrics across answer outcomes ...")
    outcome_sections: List[str] = [
        "## Outcome Comparison",
        "",
        textwrap.fill(
            "To contextualise complexity, the table below contrasts mean metric values between records whose evaluated "
            "graph matched the gold answer and those that did not.",
            width=100,
        ),
        "",
    ]

    outcome_rows: List[str] = []
    insufficient: List[str] = []
    for qty in quantities:
        match_vals = safe_float_list(
            rec.get(qty.key)
            for rec in records
            if rec.get("answer_match") is True
        )
        nonmatch_vals = safe_float_list(
            rec.get(qty.key)
            for rec in records
            if rec.get("answer_match") is False
        )
        if match_vals and nonmatch_vals:
            mean_match = float(np.mean(match_vals))
            mean_nonmatch = float(np.mean(nonmatch_vals))
            diff = mean_match - mean_nonmatch
            outcome_rows.append(
                "| {title} | {match} | {nonmatch} | {diff} |".format(
                    title=qty.title,
                    match=format_float(mean_match),
                    nonmatch=format_float(mean_nonmatch),
                    diff=format_float(diff),
                )
            )
        else:
            insufficient.append(qty.title)

    if outcome_rows:
        outcome_sections.append("| Metric | Mean (match) | Mean (non-match) | Δ |")
        outcome_sections.append("|---|---:|---:|---:|")
        outcome_sections.extend(outcome_rows)
        outcome_sections.append("")
    if insufficient:
        outcome_sections.append(
            textwrap.fill(
                "Metrics without enough matched and mismatched examples for comparison: "
                + ", ".join(insufficient)
                + ".",
                width=100,
            )
        )
        outcome_sections.append("")

    quantity_sections.extend(outcome_sections)

    report_path = os.path.join(report_dir, "report.md")
    log(f"Rendering markdown report to {report_path}")
    render_markdown(
        report_path=report_path,
        arrow_path=arrow_path,
        total_records=total_records,
        summary_lines=summary_lines,
        stat_rows=stat_rows,
        quantity_sections=quantity_sections,
    )

    artifacts = [
        ("Report", report_path),
    ]

    # Collect generated images for quick visibility.
    if os.path.isdir(images_dir):
        image_files = sorted(
            os.path.join(images_dir, name)
            for name in os.listdir(images_dir)
            if name.lower().endswith(".png")
        )
        for img in image_files:
            artifacts.append(("Image", img))

    log("Markdown report and image assets generated.")
    print("Analysis complete. Artifacts:")
    for label, path in artifacts:
        print(f"  - {label}: {path}")


if __name__ == "__main__":
    main()
