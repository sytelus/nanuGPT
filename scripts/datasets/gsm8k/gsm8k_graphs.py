
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GSM8K → Computational Graphs (DAG) via Azure OpenAI (GPT-5)
==========================================================

What this script does
---------------------
1) Downloads the GSM8K dataset from Hugging Face ("main" config; train & test splits).
2) For each problem, asks a GPT-5 model (via Azure OpenAI) to emit a *computational graph*
   as JSON using a constrained function-call ("tools") schema.
3) Validates:
   - JSON and schema shape
   - node/edge consistency
   - DAG acyclicity
4) Evaluates the graph to compute the final numeric output (when possible), compares it to
   the gold answer, and computes graph statistics (num nodes/edges, height, max width, ops).
5) Writes results to a restartable cache (JSONL) and, at the end (or on demand), emits a
   Hugging Face Arrow dataset on disk with the original columns + graph + metadata.

Key features for robustness
---------------------------
- Exponential backoff + jitter for transient API failures (rate limiting, timeouts).
- Content-repair loop if the model returns invalid JSON/graph (up to N attempts).
- Parallelism (async) with a bounded concurrency semaphore.
- Resumable: if interrupted, re-run with the same output directory to pick up where you left off.
- Periodic progress logs: speed, ETA, success/error counts, retry counts.

Prerequisites
-------------
pip install -U:
  openai>=1.40.0
  datasets>=2.20.0
  orjson>=3.10.0
  tenacity>=8.5.0
  rich>=13.7.0
  pyarrow>=16.0.0

Environment / Config
--------------------
You must provide Azure OpenAI details either via environment variables or command-line flags:
  AZURE_OPENAI_ENDPOINT (e.g. https://<your-resource>.openai.azure.com/)
  AZURE_OPENAI_API_KEY
  OPENAI_API_VERSION (e.g. 2024-06-01)  # You can override via --api-version
  AZURE_OPENAI_DEPLOYMENT (the deployment name for your GPT-5 (or compatible) chat model)

Example
-------
python gsm8k_graphs_pipeline.py \
  --outdir ./out/gsm8k_graphs \
  --dataset gsm8k --config main \
  --splits train test \
  --max-workers 8 \
  --dry-run 50

At the end, you can build the Arrow dataset:
python gsm8k_graphs_pipeline.py --outdir ./out/gsm8k_graphs --build-arrow

Notes
-----
- This script targets Azure OpenAI Chat Completions API with "tools" (function calling).
- If your Azure deployment or API version differs, set them via flags or environment.
- The validator/evaluator supports typical middle-school math ops; unknown ops won't crash
  validation but will likely prevent evaluation; the script records those gracefully.

"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import datetime as dt
import functools
import itertools
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import orjson
from tenacity import retry, stop_after_attempt, wait_exponential_jitter, retry_if_exception_type

# Rich for pretty progress and logs
from rich.console import Console
from rich.table import Table
from rich.progress import (
    Progress, SpinnerColumn, BarColumn, TimeElapsedColumn,
    TimeRemainingColumn, TextColumn, MofNCompleteColumn
)
from rich.logging import RichHandler
import logging

# Hugging Face datasets / Arrow
from datasets import load_dataset, Dataset, DatasetDict
import pyarrow as pa
import pyarrow.dataset as pads
from nanugpt import utils


# -----------------------------
# Configuration
# -----------------------------

@dataclasses.dataclass
class Config:
    # Azure OpenAI
    azure_endpoint: str = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
    api_key: str = os.environ.get("AZURE_OPENAI_API_KEY", "")
    deployment: str = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "")
    api_version: str = os.environ.get("OPENAI_API_VERSION", "2024-06-01")

    # Run controls
    # Final output directory. If not provided, we derive it at runtime as:
    #   os.environ.get('OUT_DIR', '~/out_dir') + '/gsm8k_graphs'
    out_dir: Optional[str] = None
    dataset_name: str = "gsm8k"
    dataset_config: str = "main"
    splits: Tuple[str, ...] = ("train", "test")
    max_workers: int = 8
    request_timeout_s: Optional[int] = None
    network_max_retries: int = 8
    content_max_attempts: int = 3   # model "repair" attempts for invalid graphs
    dry_run: int = 0                # if >0, only process this many per split

    # Prompt knobs
    temperature: Optional[float] = None
    seed: Optional[int] = 7

    # Build Arrow dataset from cached jsonl even without calling the API
    build_arrow_only: bool = False


# -----------------------------
# Azure OpenAI client (async)
# -----------------------------

from openai import AsyncAzureOpenAI


class AOAI:
    """Thin async wrapper over OpenAI Python lib configured for Azure endpoints."""
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        # Configure Azure client explicitly for clarity.
        self.client = AsyncAzureOpenAI(
            api_key=cfg.api_key,
            azure_endpoint=cfg.azure_endpoint,
            api_version=cfg.api_version,
        )
        self.deployment = cfg.deployment
        self.api_version = cfg.api_version

    async def chat(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None,
                   tool_choice: Optional[Dict[str, Any]] = None, timeout_s: Optional[int] = None,
                   temperature: Optional[float] = None, seed: Optional[int] = None) -> Dict[str, Any]:

        # The openai client uses kwargs for Azure specifics.
        kwargs: Dict[str, Any] = dict(
            model=self.deployment,
            messages=messages,
        )
        if temperature is not None:
            kwargs["temperature"] = temperature
        if tools is not None:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        if seed is not None:
            kwargs["seed"] = seed

        if timeout_s is not None:
            resp = await asyncio.wait_for(
                self.client.chat.completions.create(**kwargs),
                timeout=timeout_s
            )
        else:
            resp = await self.client.chat.completions.create(**kwargs)
        return resp.to_dict_recursive() if hasattr(resp, "to_dict_recursive") else resp


# -----------------------------
# Prompt: system + tool schema
# -----------------------------

ALLOWED_OPS = [
    # literals / IO
    "Input", "Const",
    # arithmetic
    "Add", "Sub", "Mul", "Div", "Pow", "Neg", "Abs",
    # aggregates / statistics
    "Sum", "Avg", "Median", "Min", "Max",
    # percentages
    "PercentOf", "IncreaseByPercent", "DecreaseByPercent",
    # integer / misc
    "Floor", "Ceil", "Round", "Mod", "FloorDiv",
    # comparisons / control (rare; avoid unless needed)
    "Equal", "Less", "Greater", "If",
]

def tool_schema() -> List[Dict[str, Any]]:
    """Function (tools) schema to force the model to emit a well-formed graph."""
    return [{
        "type": "function",
        "function": {
            "name": "emit_graph",
            "description": "Return a computational graph (DAG) for the given middle-school math word problem.",
            "parameters": {
                "type": "object",
                "properties": {
                    "nodes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string", "description": "Unique short identifier for the node (use a-z, 0-9, _)."},
                                "op": {"type": "string", "description": f"One of: {', '.join(ALLOWED_OPS)}. Prefer these canonical names."},
                                "inputs": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "IDs of upstream nodes for this operation, in left-to-right order where relevant. Omit for Input/Const."
                                },
                                "value": {
                                    "type": ["number", "string", "null"],
                                    "description": "Literal value when op is Input or Const; often a number. For Input, include the given value from the problem text if present."
                                },
                                "units": {"type": ["string", "null"], "description": "Optional units label, e.g., 'gumballs', 'minutes'."},
                                "type": {"type": ["string", "null"], "description": "Optional type, e.g., 'scalar' (default)."}
                            },
                            "required": ["id", "op"],
                            "additionalProperties": False
                        }
                    },
                    "edges": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "minItems": 2,
                            "maxItems": 2,
                            "items": {"type": "string"}
                        },
                        "description": "Edge list [src, dst]. These should correspond exactly to 'inputs' arrays on dst nodes. You may omit 'edges' and it will be derived."
                    },
                    "output": {"type": "string", "description": "The id of the final output node."},
                    "notes": {"type": "string", "description": "Short natural-language note describing how the graph maps to the problem text."}
                },
                "required": ["nodes", "output"],
                "additionalProperties": False
            }
        }
    }]


SYSTEM_PROMPT = """\
You are a meticulous graph builder for middle-school math word problems.

Task:
Given a problem statement, produce a computational graph (a DAG) capturing the exact arithmetic.
Use a small, explicit set of operations. Do NOT solve using algebra in one step—break it into
simple arithmetic consistent with the text.

Rules:
- Use concise node ids (a, b, total, k2, etc.).
- Put any constants like “twice”, “4 less than”, percentages, etc., as Const nodes.
- Prefer canonical ops: Input, Const, Add, Sub, Mul, Div, Pow, Neg, Abs, Sum, Avg, Median, Min, Max,
  PercentOf, IncreaseByPercent, DecreaseByPercent, Floor, Ceil, Round, Mod, FloorDiv, Equal, Less, Greater, If.
- For “x percent of y”, use PercentOf with inputs [percent, y], where percent is numeric like 15 for 15%.
- For “increased by p%” or “decreased by p%”, use IncreaseByPercent / DecreaseByPercent with inputs [base, p].
- Include Input nodes for any known quantities directly from the problem text (with 'value').
- Include units when obvious.
- Ensure the graph is acyclic and each referenced input id exists.
- Prefer composing multi-input Add/Mul instead of chaining two-step adds/mults when it’s clearer.
- Set 'output' to the node representing the final numeric answer requested.

Response format:
Return ONLY via the function call 'emit_graph' with arguments matching the schema.
If the problem is underspecified or non-numeric, still return a best-effort DAG capturing the numeric parts.
"""


def user_prompt(question: str) -> List[Dict[str, Any]]:
    """Build messages with the system content and the problem statement for the user role."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Problem:\n{question}\n\nReturn only the structured graph via the emit_graph function."
        }
    ]


REPAIR_SYSTEM_PROMPT = """\
You previously returned an invalid graph. Fix it strictly according to the validator errors below.
Return ONLY via the 'emit_graph' function. Use only the allowed ops. Keep ids stable where possible.
"""


# -----------------------------
# Graph validation, evaluation, stats
# -----------------------------

class GraphError(Exception):
    pass


def json_dumps(obj: Any) -> str:
    return orjson.dumps(obj, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS).decode("utf-8")


def parse_gold_answer(ans: str) -> Optional[float]:
    """
    GSM8K answers often end with '#### 42'. Extract the trailing number.
    If not found, fallback to the last number in the string.
    """
    if not ans:
        return None
    m = re.search(r'####\s*([-+]?\d+(?:\.\d+)?)\s*$', ans.strip())
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass
    # fallback: last number
    nums = re.findall(r'[-+]?\d+(?:\.\d+)?', ans)
    if nums:
        try:
            return float(nums[-1])
        except Exception:
            return None
    return None


def build_edges_from_inputs(nodes: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    edges = []
    idset = {n["id"] for n in nodes if "id" in n}
    for n in nodes:
        ins = n.get("inputs") or []
        for src in ins:
            if src in idset:
                edges.append((src, n["id"]))
    return edges


def topo_sort(nodes: List[Dict[str, Any]]) -> Tuple[List[str], Dict[str, int]]:
    """Kahn's algorithm. Returns topological order and level mapping (depth)."""
    id_to_node = {n["id"]: n for n in nodes}
    indeg = defaultdict(int)
    adj = defaultdict(list)
    for n in nodes:
        for src in n.get("inputs") or []:
            indeg[n["id"]] += 1
            adj[src].append(n["id"])
    q = [nid for nid in id_to_node if indeg[nid] == 0]
    order = []
    level = {nid: 0 for nid in id_to_node}

    from collections import deque
    dq = deque(q)
    while dq:
        u = dq.popleft()
        order.append(u)
        for v in adj[u]:
            level[v] = max(level[v], level[u] + 1)
            indeg[v] -= 1
            if indeg[v] == 0:
                dq.append(v)

    if len(order) != len(nodes):
        raise GraphError("Cycle detected or missing nodes in topological sort.")
    return order, level


def compute_stats(nodes: List[Dict[str, Any]], edges: List[Tuple[str, str]], output_id: str) -> Dict[str, Any]:
    order, level = topo_sort(nodes)
    width_by_level = Counter(level.values())
    max_width = max(width_by_level.values()) if width_by_level else 0
    height = (max(level.values()) + 1) if level else 0
    ops = sorted({n.get("op", "") for n in nodes if n.get("op")})
    return {
        "num_nodes": len(nodes),
        "num_edges": len(edges),
        "max_width": int(max_width),
        "height": int(height),
        "ops": ops,
        "levels": {k: v for k, v in sorted(width_by_level.items())},
        "topo_ok": True,
    }


def validate_graph(graph: Dict[str, Any]) -> Tuple[bool, List[str], Dict[str, Any]]:
    errors: List[str] = []
    if not isinstance(graph, dict):
        return False, ["Graph is not a dict"], {}

    nodes = graph.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        errors.append("Missing or empty 'nodes' array.")

    output_id = graph.get("output")
    if not isinstance(output_id, str) or not output_id:
        errors.append("Missing 'output' id string.")

    id_to_node: Dict[str, Dict[str, Any]] = {}
    if isinstance(nodes, list):
        for n in nodes:
            if not isinstance(n, dict):
                errors.append("Node is not an object.")
                continue
            nid = n.get("id")
            op = n.get("op")
            if not isinstance(nid, str) or not nid:
                errors.append("A node is missing 'id'.")
                continue
            if nid in id_to_node:
                errors.append(f"Duplicate node id: {nid}")
                continue
            id_to_node[nid] = n
            if not isinstance(op, str) or not op:
                errors.append(f"Node {nid} missing 'op'.")
            if op in ("Input", "Const"):
                if "value" not in n:
                    errors.append(f"Node {nid} (op={op}) missing 'value'.")
            else:
                ins = n.get("inputs")
                if ins is None or not isinstance(ins, list) or len(ins) < 1:
                    errors.append(f"Node {nid} (op={op}) must have non-empty 'inputs'.")

    # Assemble edges
    if isinstance(nodes, list):
        derived_edges = build_edges_from_inputs(nodes)
    else:
        derived_edges = []
    provided_edges = graph.get("edges")
    if provided_edges is None:
        edges = derived_edges
    else:
        # Basic validation of provided edges structure
        edges = []
        if not isinstance(provided_edges, list):
            errors.append("'edges' must be a list of [src, dst].")
        else:
            for e in provided_edges:
                if not (isinstance(e, list) or isinstance(e, tuple)) or len(e) != 2:
                    errors.append(f"Edge {e} must be a [src, dst] pair.")
                else:
                    edges.append((str(e[0]), str(e[1])))
        # Optional: ensure provided exactly equals derived
        if set(edges) != set(derived_edges):
            errors.append("Provided 'edges' do not match the edges implied by 'inputs'.")

    # Verify all edge endpoints exist
    idset = set(id_to_node.keys())
    for s, d in edges:
        if s not in idset:
            errors.append(f"Edge src id not found: {s}")
        if d not in idset:
            errors.append(f"Edge dst id not found: {d}")

    # Output existence
    if isinstance(output_id, str) and output_id not in idset:
        errors.append(f"'output' id not present among nodes: {output_id}")

    topo_ok = False
    stats = {}
    if not errors and nodes:
        try:
            stats = compute_stats(nodes, edges, output_id)
            topo_ok = True
        except GraphError as ge:
            errors.append(str(ge))

    valid = len(errors) == 0 and topo_ok
    return valid, errors, stats


def _as_number(x: Any) -> float:
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        # allow numeric strings like "15" or "15.0"
        try:
            return float(x)
        except Exception:
            pass
    raise GraphError(f"Non-numeric value encountered: {x!r}")


def evaluate_graph(graph: Dict[str, Any]) -> Tuple[bool, Optional[float], Optional[str]]:
    """
    Evaluate the graph to compute a numeric output.
    Returns (success, value, error_message_or_None).
    """
    nodes = graph.get("nodes") or []
    id_to_node = {n["id"]: n for n in nodes}
    order, level = topo_sort(nodes)

    values: Dict[str, float] = {}

    for nid in order:
        n = id_to_node[nid]
        op = n.get("op")
        ins = [values[i] for i in (n.get("inputs") or [])]  # downstream nodes see upstream computed
        try:
            if op in ("Input", "Const"):
                values[nid] = _as_number(n.get("value"))
            elif op == "Add" or op == "Sum":
                if not ins:
                    raise GraphError("Add/Sum with no inputs.")
                values[nid] = float(sum(ins))
            elif op == "Sub":
                if len(ins) < 2:
                    raise GraphError("Sub requires at least 2 inputs.")
                values[nid] = float(ins[0] - sum(ins[1:]))
            elif op == "Mul":
                if not ins:
                    raise GraphError("Mul with no inputs.")
                prod = 1.0
                for x in ins:
                    prod *= x
                values[nid] = float(prod)
            elif op == "Div":
                if len(ins) != 2:
                    raise GraphError("Div requires exactly 2 inputs.")
                if ins[1] == 0:
                    raise GraphError("Division by zero.")
                values[nid] = float(ins[0] / ins[1])
            elif op == "Pow":
                if len(ins) != 2:
                    raise GraphError("Pow requires exactly 2 inputs.")
                values[nid] = float(pow(ins[0], ins[1]))
            elif op == "Neg":
                if len(ins) != 1:
                    raise GraphError("Neg requires exactly 1 input.")
                values[nid] = float(-ins[0])
            elif op == "Abs":
                if len(ins) != 1:
                    raise GraphError("Abs requires exactly 1 input.")
                values[nid] = float(abs(ins[0]))
            elif op == "Avg":
                if not ins:
                    raise GraphError("Avg with no inputs.")
                values[nid] = float(sum(ins) / len(ins))
            elif op == "Median":
                if not ins:
                    raise GraphError("Median with no inputs.")
                s = sorted(ins)
                m = len(s) // 2
                if len(s) % 2 == 1:
                    values[nid] = float(s[m])
                else:
                    values[nid] = float((s[m - 1] + s[m]) / 2.0)
            elif op == "Min":
                if not ins:
                    raise GraphError("Min with no inputs.")
                values[nid] = float(min(ins))
            elif op == "Max":
                if not ins:
                    raise GraphError("Max with no inputs.")
                values[nid] = float(max(ins))
            elif op == "PercentOf":
                if len(ins) != 2:
                    raise GraphError("PercentOf requires [percent, value].")
                values[nid] = float(ins[0] * ins[1] / 100.0)
            elif op == "IncreaseByPercent":
                if len(ins) != 2:
                    raise GraphError("IncreaseByPercent requires [base, percent].")
                values[nid] = float(ins[0] * (1.0 + ins[1] / 100.0))
            elif op == "DecreaseByPercent":
                if len(ins) != 2:
                    raise GraphError("DecreaseByPercent requires [base, percent].")
                values[nid] = float(ins[0] * (1.0 - ins[1] / 100.0))
            elif op == "Floor":
                if len(ins) != 1:
                    raise GraphError("Floor requires 1 input.")
                values[nid] = float(math.floor(ins[0]))
            elif op == "Ceil":
                if len(ins) != 1:
                    raise GraphError("Ceil requires 1 input.")
                values[nid] = float(math.ceil(ins[0]))
            elif op == "Round":
                if len(ins) != 1:
                    raise GraphError("Round requires 1 input.")
                values[nid] = float(round(ins[0]))
            elif op == "Mod":
                if len(ins) != 2:
                    raise GraphError("Mod requires 2 inputs.")
                if ins[1] == 0:
                    raise GraphError("Modulo by zero.")
                values[nid] = float(ins[0] % ins[1])
            elif op == "FloorDiv":
                if len(ins) != 2:
                    raise GraphError("FloorDiv requires 2 inputs.")
                if ins[1] == 0:
                    raise GraphError("Division by zero.")
                values[nid] = float(ins[0] // ins[1])
            elif op in ("Equal", "Less", "Greater", "If"):
                # Logical ops not used for numeric final answers; skip evaluation by treating as error
                raise GraphError(f"Logical op {op} not supported in numeric evaluation.")
            else:
                raise GraphError(f"Unknown op: {op}")
        except GraphError as ge:
            return False, None, f"Error at node {nid}: {ge}"

    output_id = graph.get("output")
    if output_id not in values:
        return False, None, f"Output id {output_id!r} has no value."
    return True, values[output_id], None


# -----------------------------
# Azure call with robust network retries
# -----------------------------

class TransientNetworkError(Exception):
    pass


def _is_transient_exception(exc: BaseException) -> bool:
    text = str(exc).lower()
    for needle in ["timeout", "temporarily unavailable", "rate limit", "429", "connection reset",
                   "service unavailable", "gateway timeout", "502", "503", "504", "dns", "retry"]:
        if needle in text:
            return True
    return False


def network_retry_decorator(max_attempts: int):
    return retry(
        reraise=True,
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential_jitter(initial=1, max=30),
        retry=retry_if_exception_type((TransientNetworkError, asyncio.TimeoutError))
    )


# -----------------------------
# Orchestration
# -----------------------------

class Runner:
    def __init__(self, cfg: Config, console: Console) -> None:
        self.cfg = cfg
        self.console = console
        self.aoai = AOAI(cfg)
        self.sem = asyncio.Semaphore(cfg.max_workers)
        self.start_time = time.time()
        self.print_lock = asyncio.Lock()

        # cache dirs
        if not cfg.out_dir or not str(cfg.out_dir).strip():
            base_out = os.environ.get("OUT_DIR", os.path.expanduser("~/out_dir"))
            final_out = os.path.join(base_out, "gsm8k_graphs")
        else:
            # Respect explicit CLI value as the final directory
            final_out = cfg.out_dir

        self.out_dir = utils.full_path(final_out, create=True)
        self.cache_dir = os.path.join(self.out_dir, "cache")
        self.logs_dir = os.path.join(self.out_dir, "logs")
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

        # logging
        log_path = os.path.join(self.logs_dir, f"run_{int(self.start_time)}.log")
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(console=self.console, rich_tracebacks=True),
                      logging.FileHandler(log_path, encoding="utf-8")]
        )
        self.log = logging.getLogger("gsm8k-graphs")

        # progress counters
        self.total = 0
        self.done = 0
        self.success = 0
        self.content_fail = 0
        self.net_fail = 0
        self.repaired = 0
        self.bytes_written = 0
        self._last_stats = (-1, -1, -1, -1, -1)

    # ---------- printing helpers ----------
    async def _print_worker(self, rid: str, message: str, style: Optional[str] = None) -> None:
        async with self.print_lock:
            prefix = f"[{rid}]"
            if style:
                self.console.print(f"{prefix} {message}", style=style)
            else:
                self.console.print(f"{prefix} {message}")

    async def _print_main(self, message: str) -> None:
        async with self.print_lock:
            self.console.print(message, style="bold cyan")

    def _eta(self) -> str:
        elapsed = max(time.time() - self.start_time, 1e-6)
        speed = self.done / elapsed if self.done > 0 else 0.0
        remaining = max(self.total - self.done, 0)
        eta_s = (remaining / speed) if speed > 0 else float("inf")
        if not math.isfinite(eta_s):
            return "ETA: ∞"
        m, s = divmod(int(eta_s), 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"ETA≈{h}h {m}m"
        return f"ETA≈{m}m {s}s"

    async def _report_progress(self) -> None:
        stats = (self.done, self.success, self.content_fail, self.net_fail, self.repaired)
        if stats == self._last_stats:
            return
        self._last_stats = stats
        await self._print_main(
            f"progress: {self.done}/{self.total} | success={self.success} | "
            f"content_fail={self.content_fail} | net_fail={self.net_fail} | repaired={self.repaired} | {self._eta()}"
        )

    # ---------- persistence helpers ----------
    def _split_dir(self, split: str) -> str:
        d = os.path.join(self.cache_dir, split)
        os.makedirs(d, exist_ok=True)
        return d

    def _jsonl_path(self, split: str) -> str:
        return os.path.join(self.cache_dir, f"{split}.jsonl")

    def _record_path(self, split: str, rid: str) -> str:
        return os.path.join(self._split_dir(split), f"{rid}.json")

    def already_done_ids(self, split: str) -> set:
        """Return set of ids that already have a cached JSON file."""
        d = self._split_dir(split)
        return {fn[:-5] for fn in os.listdir(d) if fn.endswith(".json")}

    def save_record(self, split: str, rid: str, obj: Dict[str, Any]) -> None:
        p = self._record_path(split, rid)
        tmp = p + ".tmp"
        with open(tmp, "wb") as f:
            f.write(orjson.dumps(obj))
        os.replace(tmp, p)
        self.bytes_written += os.path.getsize(p)

    # ---------- model call & repair ----------
    async def call_model_once(self, messages: List[Dict[str, Any]], is_repair: bool = False, rid: Optional[str] = None) -> Dict[str, Any]:
        attempts = 0
        delay = 1.0
        while True:
            attempts += 1
            try:
                if rid:
                    await self._print_worker(rid, f"API call attempt {attempts}{' (repair)' if is_repair else ''} ...")
                resp = await self.aoai.chat(
                    messages=messages,
                    tools=tool_schema(),
                    tool_choice={"type": "function", "function": {"name": "emit_graph"}},
                    timeout_s=self.cfg.request_timeout_s,
                    temperature=self.cfg.temperature,
                    seed=self.cfg.seed,
                )
                if rid:
                    await self._print_worker(rid, f"API call attempt {attempts} succeeded", style="green")
                return resp
            except Exception as e:
                # Transient?
                if _is_transient_exception(e) and attempts < self.cfg.network_max_retries:
                    if rid:
                        await self._print_worker(rid, f"Transient API error: {e}; retrying in {delay:.1f}s", style="yellow")
                    await asyncio.sleep(delay + random.random() * 0.5 * delay)
                    delay = min(delay * 2.0, 30.0)
                    continue
                # Non-transient or out of retries
                if _is_transient_exception(e):
                    if rid:
                        await self._print_worker(rid, f"API failed after {attempts} attempts: {e}", style="red")
                    raise TransientNetworkError(str(e))
                if rid:
                    await self._print_worker(rid, f"API error (non-retryable): {e}", style="red")
                raise

    def extract_graph(self, resp: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str], Dict[str, Any]]:
        """Try to extract the function-call arguments or JSON content as graph."""
        raw = None
        try:
            # Newer SDKs
            choices = resp.get("choices") or []
            if choices and "message" in choices[0]:
                msg = choices[0]["message"]
                tool_calls = msg.get("tool_calls") or []
                if tool_calls and "function" in tool_calls[0]:
                    fn = tool_calls[0]["function"]
                    raw = fn.get("arguments")
                    if raw:
                        graph = orjson.loads(raw)
                        return graph, None, resp
                # Fallback to message content as JSON
                content = msg.get("content")
                if isinstance(content, str):
                    raw = content.strip()
                    # Try to extract a JSON code block if present
                    m = re.search(r"\{.*\}", raw, flags=re.S)
                    if m:
                        raw = m.group(0)
                    graph = orjson.loads(raw)
                    return graph, None, resp
        except Exception as e:
            return None, f"Failed to parse model output as JSON: {e}", resp
        return None, "No function call or JSON content found.", resp

    async def build_or_repair_graph(self, question: str, rid: Optional[str] = None) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any], List[str], int]:
        """Attempt initial build; if invalid, send one or two repair attempts with validator feedback."""
        messages = user_prompt(question)
        errors_all: List[str] = []
        attempts = 0

        # Initial call
        attempts += 1
        if rid:
            await self._print_worker(rid, "Building graph: initial API call")
        resp = await self.call_model_once(messages, rid=rid)
        graph, parse_err, raw_resp = self.extract_graph(resp)
        if graph is None:
            if rid:
                await self._print_worker(rid, f"Unexpected API return: {parse_err}", style="red")
            errors_all.append(parse_err or "Parse error")
        else:
            valid, errors, _ = validate_graph(graph)
            if valid:
                if rid:
                    await self._print_worker(rid, "Initial graph valid", style="green")
                return graph, raw_resp, [], attempts
            errors_all.extend(errors)
            if rid:
                await self._print_worker(rid, f"Initial graph invalid: {len(errors)} error(s)", style="yellow")

        # Repair attempts
        for _ in range(self.cfg.content_max_attempts - 1):
            attempts += 1
            repair_messages = [
                {"role": "system", "content": REPAIR_SYSTEM_PROMPT},
                {"role": "user", "content": "Validator errors:\n" + "\n".join(f"- {e}" for e in errors_all)},
            ]
            if rid:
                await self._print_worker(rid, f"Repair attempt {attempts - 1}")
            resp = await self.call_model_once(repair_messages, is_repair=True, rid=rid)
            graph, parse_err, raw_resp = self.extract_graph(resp)
            if graph is None:
                if rid:
                    await self._print_worker(rid, f"Unexpected API return on repair: {parse_err}", style="red")
                errors_all.append(parse_err or "Parse error")
                continue
            valid, errors, _ = validate_graph(graph)
            if valid:
                self.repaired += 1
                await self._report_progress()
                if rid:
                    await self._print_worker(rid, "Graph repaired successfully", style="green")
                return graph, raw_resp, [], attempts
            errors_all.extend(errors)
            if rid:
                await self._print_worker(rid, f"Repair failed: {len(errors)} validation error(s)", style="yellow")

        return None, raw_resp if 'raw_resp' in locals() else {}, errors_all, attempts

    # ---------- per-record processing ----------
    async def process_one(self, split: str, rid: str, question: str, answer: str) -> Dict[str, Any]:
        try:
            await self._print_worker(rid, "Starting problem")
            graph, raw, errors, attempts = await self.build_or_repair_graph(question, rid=rid)
        except TransientNetworkError as ne:
            self.net_fail += 1
            await self._report_progress()
            await self._print_worker(rid, f"NetworkError: {ne}", style="red")
            return {
                "id": rid,
                "split": split,
                "question": question,
                "answer": answer,
                "error": f"NetworkError: {ne}",
                "attempts": 0,
                "ts": dt.datetime.utcnow().isoformat()
            }

        record: Dict[str, Any] = {
            "id": rid,
            "split": split,
            "question": question,
            "answer": answer,
            "ts": dt.datetime.utcnow().isoformat(),
            "attempts": attempts,
        }

        if graph is None:
            self.content_fail += 1
            await self._report_progress()
            await self._print_worker(rid, f"Bad graph; {len(errors)} error(s)", style="red")
            record.update({
                "graph": None,
                "graph_valid": False,
                "validation_errors": errors,
                "raw_response": raw,
            })
            return record

        valid, verrors, stats = validate_graph(graph)
        record.update({
            "graph": graph,
            "graph_valid": bool(valid),
            "validation_errors": verrors,
            "raw_response": raw,
        })

        # evaluation
        eval_ok, value, eval_err = (False, None, None)
        if valid:
            try:
                eval_ok, value, eval_err = evaluate_graph(graph)
            except Exception as e:
                eval_ok, value, eval_err = False, None, str(e)

        gold = parse_gold_answer(answer)
        match = (eval_ok and gold is not None and value is not None and abs(value - gold) < 1e-6)

        record.update({
            "eval_ok": bool(eval_ok),
            "eval_value": value,
            "eval_error": eval_err,
            "gold_value": gold,
            "answer_match": bool(match),
            "stats": stats,
        })

        self.success += 1
        await self._report_progress()
        await self._print_worker(rid, f"Done. valid={bool(valid)} match={bool(match)} attempts={attempts}", style="green" if valid else "yellow")
        return record

    # ---------- split processing ----------
    async def process_split(self, split: str) -> None:
        self.console.rule(f"[bold]Processing split: {split}")
        ds = load_dataset(self.cfg.dataset_name, self.cfg.dataset_config, split=split)
        done = self.already_done_ids(split)
        n_total = len(ds) if self.cfg.dry_run <= 0 else min(self.cfg.dry_run, len(ds))
        self.total += n_total
        self.log.info(f"{split}: {n_total} total; {len(done)} already cached; will skip cached.")

        async def worker(idx: int, row: Dict[str, Any]):
            rid = f"{split}_{idx:05d}"
            if rid in done:
                await self._print_worker(rid, "Skipping (cached)", style="dim")
                return  # already processed
            async with self.sem:
                await self._print_worker(rid, "Acquired worker slot", style="blue")
                rec = await self.process_one(split, rid, row["question"], row["answer"])
                self.save_record(split, rid, rec)
                self.done += 1
                await self._report_progress()

        tasks = []
        for idx, row in enumerate(ds):
            if self.cfg.dry_run > 0 and idx >= self.cfg.dry_run:
                break
            rid = f"{split}_{idx:05d}"
            if rid in done:
                self.done += 1  # count towards progress
                await self._report_progress()
                continue
            tasks.append(asyncio.create_task(worker(idx, row)))

        # Await all to raise exceptions if any
        await asyncio.gather(*tasks, return_exceptions=True)
        await self._print_main(f"[green]Finished split {split}[/green]")

    # ---------- Arrow dataset build ----------
    def build_arrow(self) -> None:
        """Read cached JSONs and build a HF Dataset saved to disk."""
        self.console.rule("[bold]Building Arrow dataset from cache")

        # Load original splits to align (for reproducibility)
        result_rows = []
        for split in self.cfg.splits:
            split_dir = self._split_dir(split)
            files = [os.path.join(split_dir, fn) for fn in os.listdir(split_dir) if fn.endswith(".json")]
            for p in sorted(files):
                with open(p, "rb") as f:
                    rec = orjson.loads(f.read())
                # Normalize for HF datasets (ensure pure Python types)
                out = {
                    "id": rec.get("id"),
                    "split": rec.get("split"),
                    "question": rec.get("question"),
                    "answer": rec.get("answer"),
                    "graph_json": json_dumps(rec.get("graph")) if rec.get("graph") is not None else None,
                    "graph_valid": bool(rec.get("graph_valid")),
                    "validation_errors": rec.get("validation_errors") or [],
                    "eval_ok": bool(rec.get("eval_ok")),
                    "eval_value": float(rec["eval_value"]) if rec.get("eval_value") is not None else None,
                    "gold_value": float(rec["gold_value"]) if rec.get("gold_value") is not None else None,
                    "answer_match": bool(rec.get("answer_match")),
                    "ops": (rec.get("stats") or {}).get("ops") or [],
                    "num_nodes": (rec.get("stats") or {}).get("num_nodes"),
                    "num_edges": (rec.get("stats") or {}).get("num_edges"),
                    "max_width": (rec.get("stats") or {}).get("max_width"),
                    "height": (rec.get("stats") or {}).get("height"),
                    "levels": [f"{k}:{v}" for k, v in ((rec.get('stats') or {}).get('levels') or {}).items()],
                    "attempts": int(rec.get("attempts") or 0),
                    "ts": rec.get("ts"),
                    "raw_response_json": json_dumps(rec.get("raw_response")) if rec.get("raw_response") else None,
                }
                result_rows.append(out)

        if not result_rows:
            self.console.print("[yellow]No cached records found. Nothing to build.[/yellow]")
            return

        ds = Dataset.from_list(result_rows)
        save_path = os.path.join(self.out_dir, "arrow_dataset")
        ds.save_to_disk(save_path)
        self.console.print(f"[green]Saved Arrow dataset to: {save_path}[/green]")

        # Also write a summary table
        ops_flat = list(itertools.chain.from_iterable(r["ops"] for r in result_rows))
        ops_counts = Counter(ops_flat)
        total_graphs = len(result_rows)
        valid_graphs = sum(1 for r in result_rows if r["graph_valid"])
        eval_ok = sum(1 for r in result_rows if r["eval_ok"])
        match = sum(1 for r in result_rows if r["answer_match"])

        tbl = Table(title="Summary")
        tbl.add_column("Metric")
        tbl.add_column("Value", justify="right")
        tbl.add_row("Total records", str(total_graphs))
        tbl.add_row("Valid graphs", str(valid_graphs))
        tbl.add_row("Evaluated OK", str(eval_ok))
        tbl.add_row("Answer matches", str(match))
        tbl.add_row("Unique operators", str(len(ops_counts)))
        self.console.print(tbl)

    # ---------- main entry ----------
    async def run(self) -> None:
        if self.cfg.build_arrow_only:
            self.build_arrow()
            # Print final output directory for user clarity
            self.console.print(f"[bold green]Output directory:[/bold green] {self.out_dir}")
            return

        # Basic sanity checks
        if not self.cfg.azure_endpoint or not self.cfg.api_key or not self.cfg.deployment:
            self.console.print("[red]Azure endpoint, key, and deployment name are required.[/red]")
            self.console.print("Set env AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT, or use flags.")
            sys.exit(2)

        # Process each split
        for split in self.cfg.splits:
            await self.process_split(split)

        # Build Arrow dataset at the end as a convenience
        self.build_arrow()
        # Print final output directory for user clarity
        self.console.print(f"[bold green]Output directory:[/bold green] {self.out_dir}")


# -----------------------------
# CLI
# -----------------------------

def parse_args(argv: Optional[List[str]] = None) -> Config:
    p = argparse.ArgumentParser(description="GSM8K → computational graphs via Azure OpenAI (GPT-5).")
    # Standardize to --out_dir; keep --outdir as a backward-compatible alias
    p.add_argument("--out_dir", "--outdir", dest="out_dir", type=str, default=None,
                   help="Output directory (cache + logs + arrow). If not set, uses $OUT_DIR or ~/out_dir, with subdir 'gsm8k_graphs'.")
    p.add_argument("--dataset", type=str, default=None, help="Hugging Face dataset name (default: gsm8k).")
    p.add_argument("--config", type=str, default=None, help="Dataset config (default: main).")
    p.add_argument("--splits", type=str, nargs="+", default=None, help="Splits to process (default: train test).")
    p.add_argument("--max-workers", type=int, default=None, help="Max concurrent API calls.")
    p.add_argument("--timeout", type=int, default=None, help="Per-request timeout (seconds).")
    p.add_argument("--net-retries", type=int, default=None, help="Max network retries.")
    p.add_argument("--content-attempts", type=int, default=None, help="Max content repair attempts per record.")
    p.add_argument("--dry-run", type=int, default=None, help="If set, only process this many examples per split.")
    p.add_argument("--temperature", type=float, default=None, help="Sampling temperature.")
    p.add_argument("--seed", type=int, default=None, help="Deterministic seed for the model (if supported).")

    # Azure
    p.add_argument("--azure-endpoint", type=str, default=None, help="Azure OpenAI endpoint URL.")
    p.add_argument("--api-key", type=str, default=None, help="Azure OpenAI API key.")
    p.add_argument("--api-version", type=str, default=None, help="Azure OpenAI API version (e.g., 2024-06-01).")
    p.add_argument("--deployment", type=str, default=None, help="Azure deployment name for your chat model.")

    # Build Arrow only
    p.add_argument("--build-arrow", action="store_true", help="Only build Arrow dataset from cache; skip API calls.")

    args = p.parse_args(argv)
    cfg = Config()

    # Override from CLI
    if args.out_dir: cfg.out_dir = args.out_dir
    if args.dataset: cfg.dataset_name = args.dataset
    if args.config: cfg.dataset_config = args.config
    if args.splits: cfg.splits = tuple(args.splits)
    if args.max_workers is not None: cfg.max_workers = args.max_workers
    if args.timeout is not None: cfg.request_timeout_s = args.timeout
    if args.net_retries is not None: cfg.network_max_retries = args.net_retries
    if args.content_attempts is not None: cfg.content_max_attempts = args.content_attempts
    if args.dry_run is not None: cfg.dry_run = args.dry_run
    if args.temperature is not None: cfg.temperature = args.temperature
    if args.seed is not None: cfg.seed = args.seed

    if args.azure_endpoint: cfg.azure_endpoint = args.azure_endpoint
    if args.api_key: cfg.api_key = args.api_key
    if args.api_version: cfg.api_version = args.api_version
    if args.deployment: cfg.deployment = args.deployment

    if args.build_arrow: cfg.build_arrow_only = True

    return cfg


def main(argv: Optional[List[str]] = None) -> None:
    cfg = parse_args(argv)
    console = Console()
    runner = Runner(cfg, console)
    try:
        asyncio.run(runner.run())
    except KeyboardInterrupt:
        console.print("[red]Interrupted by user.[/red]")
    except Exception as e:
        console.print(f"[red]Fatal error:[/red] {e}")
        raise


if __name__ == "__main__":
    main()
