#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gsm8k_nprobs.py

Purpose
-------
Create a new dataset by *combining* `n_problems` randomly chosen GSM8K problems from the selected split into a single
problem whose final answer is guaranteed to match the original answer of the final selected problem.
For each attempt we:
  1) Pick `n_problems` random, distinct problems from GSM8K (train by default; `--split test` supported).
  2) Ask Azure OpenAI to *rewrite* (i.e., combine) those problems into one using a strict template.
  3) Ask Azure OpenAI to *solve* the combined problem, returning only the final numeric answer.
  4) Validate that the final numeric answer equals the GSM8K final answer of the last selected problem.
  5) If valid, append to `out_dataset/problems{n_problems}_success/<split>.jsonl` (train.jsonl by default) in GSM8K format
     (fields: {"question": <combined problem>, "answer": "#### <final_number>"}).
     We also log provenance in `out_dataset/problems{n_problems}_success/meta<split>.jsonl` (`meta.jsonl` when split=train).
  6) Otherwise, append a detailed record to `out_dataset/problems{n_problems}_fail/<split>.jsonl` (train.jsonl by default) for debugging.

The script is designed to be:
  - **Robust**: Retries with exponential backoff for Azure API calls and file writes.
  - **Restartable**: Idempotent outputs; reads existing outputs on startup and resumes until the
    target number of successes is reached.
  - **Parallel**: Uses multiple worker threads to keep the Azure API pipeline full.
  - **Observable**: Prints a live terminal dashboard (via `rich`) with per-worker stats plus a
    master row showing totals, throughput and ETA. Nonessential logs are suppressed.

Non-obvious design choices (and deviations from the user guideline)
-------------------------------------------------------------------
1) **Success format**: GSM8K "answer" usually contains chain-of-thought plus a final line
   "#### <number>". Emitting detailed reasoning could risk leaking chain-of-thought. We therefore
   write a minimal GSM8K-compatible answer string: just `"#### <number>"`. This preserves the
   corpus contract that the final answer follows `####` while avoiding unnecessary content.

2) **Provenance logging**: To fulfill the requirement to "log unique IDs" of the base problems
   in the *final output*, we add `out_dataset/problems{n_problems}_success/meta.jsonl` with, for each
   generated item, the ordered list of source problem IDs, a content hash of the combined problem, and
   timestamps. This keeps the success dataset strictly GSM8K-formatted while still shipping the
   needed debug info alongside it.

3) **Resumability**: We store outputs directly in their final files using append-only JSONL.
   On restart, we count existing successes and continue. We also maintain a `working_dir/attempts.jsonl`
  and `working_dir/seen_combos.txt` to avoid retrying identical combinations across runs. This ensures
   the script can be stopped anytime and restarted without losing progress.

4) **Answer parsing**: GSM8K answers end with `#### <final>`, but the solver may reply in various
   numeric forms (integer, decimal, or fraction). We normalize both expected and predicted answers
   to exact rational values (`fractions.Fraction`), supporting mixed numbers (e.g., `1 1/2`), plain
   fractions (e.g., `7/3`), decimals, `$` and `%` decorations, and thousands separators. This keeps
   validation strict yet robust.

5) **Minimizing spurious console noise**: We explicitly reduce the verbosity of third-party
   libraries (like `datasets`) so the rich dashboard remains readable.

Quick Start
-----------
1) Install dependencies:
   pip install --upgrade openai==1.* datasets rich

2) Set environment variables for Azure OpenAI:
   export AZURE_OPENAI_API_KEY="..."
   export AZURE_OPENAI_ENDPOINT="https://<your-resource-name>.openai.azure.com"
   export OPENAI_API_VERSION="2024-10-01-preview"   # or the version you use
   export AZURE_OPENAI_DEPLOYMENT="<your-chat-deployment-name>"

3) Optional: choose where to write output (default: ./gsm8k_{n_problems}probs in current dir)
   export OUT_DIR="/path/to/output_root"

4) Run:
   python gsm8k_nprobs.py --max_samples 200 --max_workers 8 --n_problems 3 [--split test]

Directory Layout (created under OUT_DIR/gsm8k_{n_problems}probs)
-----------------------------------------------------
in_dataset/
  gsm8k_<split>.jsonl           # input GSM8K data with sequential IDs for reference
working_dir/
  attempts<suffix>.jsonl        # every attempt appended here (suffix omitted for train)
  seen_combos<suffix>.txt       # set of processed combinations (suffix omitted for train)
out_dataset/
  problems{n_problems}_success/
    <split>.jsonl               # GSM8K-format output (question, answer); train.jsonl by default
    meta<suffix>.jsonl          # provenance: source problem IDs, hashes; meta.jsonl when split=train
  problems{n_problems}_fail/
    <split>.jsonl               # detailed diagnostics for failed attempts; train.jsonl by default

Notes
-----
- The script uses the GSM8K train split by default but accepts `--split test` when needed.
- We never persist the Azure API raw responses beyond the minimum needed; only derived, non-PII
  content is saved.
- The dashboard updates at a steady cadence; if you prefer quieter operation, add `--quiet`.

"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import os
import queue
import random
import re
import signal
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

# Third-party
try:
    from openai import AzureOpenAI
except Exception:
    AzureOpenAI = None  # We detect and error at runtime with a clear message.

try:
    from datasets import load_dataset
    from datasets.utils.logging import set_verbosity_error as hf_set_verbosity_error
except Exception:
    load_dataset = None
    hf_set_verbosity_error = None

from fractions import Fraction

try:
    from rich.console import Console, Group
    from rich.live import Live
    from rich.table import Table
    from rich.panel import Panel
    from rich.box import SIMPLE
except Exception:
    Console = None
    Live = None
    Table = None
    Panel = None
    SIMPLE = None


# Reasonable defaults; can be overridden by CLI flags
DEFAULT_MAX_SAMPLES = 2000
DEFAULT_N_PROBLEMS = 4
DEFAULT_MAX_WORKERS = 8
DEFAULT_MAX_RETRIES = 8
DEFAULT_REQ_TIMEOUT = 300.0  # 5 minutes
DEFAULT_BACKOFF_BASE = 2.0
DEFAULT_BACKOFF_JITTER = (0.2, 0.6)  # seconds
RNG_SEED = None  # set to int for reproducibility; None mixes in system entropy

# ------------------------------ Configuration ------------------------------

GRAPH_INSTRUCTIONS= [
    # n_problems = 1
    "",

    # n_problems = 2
    """- Output of Problem 1 should become input for Problem {n}.""",

    # n_problems = 3
    """- Output of Problem 1 should become input for Problem 2 as well as Problem {n}.
- Output of Problem 2 should become input for Problem {n}.""",

    # n_problems = 4
    """- Output of Problem 1 should become input for Problem 2, Problem 3 as well as Problem {n}.
- Output of Problem 2 should become input for Problem 3 as well as Problem {n}.
- Output of Problem 3 should become input for Problem {n}.""",

    # n_problems = 5
    """- Output of Problem 1 should become input for Problem 2, Problem 4 as well as Problem {n}.
- Output of Problem 2 should become input for Problem 4 as well as Problem {n}.
- Output of Problem 3 should become input for Problem 4.
- Output of Problem 4 should become input for Problem {n}.""",
]

COMBINE_PROMPT_TEMPLATE = """Consider below {n} math problems. Each of these problems have one or many input values and only one output value. Only the direct numerical values specified in the problem can be considered as input value.

We want to combine these {n} problems to form a one single problem in the following way:

{graph_instructions}
- Output of Problem {n} should become the final output of the combined problem and this output value should remain exactly same as it is in original Problem {n}.

When replacing input of a problem with output of another problem, you must make sure that final input value of the problem remains same as in original problem. To achieve this goal, you might chose to transform output value of a problem by adding some constant or multiplying with some constant to convert it before making it as an input value for the another problem.

It may happen that a problem does not enough inputs available for replacement from the output of other problems as specified above. In this case, you should combine outputs of those problems using sum operator to produce same number of different aggregated values as equal to number of inputs available in target problem. These aggregated values should be transformed as needed by adding a constant or multiplying with a constant so that it matches the original input values of the target problem. If target problem has different input values available then you should use one for each outputs of source problems.

You should keep the problem statements of each of the problem as close to the original problem statement as possible and make only the minimal number of changes required to satisfy above requirements. You can label output of each problem using a letter so that you can reference it later for the input for the another problem. Your final problem statement should not include any problem labels such as "Problem A" etc but it can include modifications such as "Let's call this value T" or any other alternative that you may find more readable, natural and faithful to original problem statement.

Using these conditions, produce the final combined problem. In your response only include this final combined problem statement in only text form and do not use any markdown or latex syntax. If for some reason, you cannot produce the combined problem then output the text "ERROR: <reason>" where <reason> is the reason why you cannot complete this task.

{problem_blocks}
"""

SOLVE_PROMPT = """Solve the following math problem. Output ONLY the final numeric answer with no words, no units, no punctuation, and no explanation. If the exact answer is a fraction, return the simplest fraction like 7/3. If the answer is a mixed number, convert it to an improper fraction. If the answer is a monetary amount, return only the number (e.g., 12.50).

Problem:
{problem}
"""

SYSTEM_COMBINE = "You are a careful math problem composer. Follow the user's instructions exactly and return only the final combined problem text."
SYSTEM_SOLVE = "You are a careful math solver. Return only the final numeric answer."



# ------------------------------ Data models ------------------------------


@dataclass
class ChatResult:
    content: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True)
class GSMItem:
    id: int
    question: str
    answer: str
    # Parsed/canonical expected final answer (if parsable), as a string and as Fraction
    expected_final_raw: Optional[str] = None
    expected_final_fraction: Optional[Fraction] = None


@dataclass
class AttemptResult:
    # Core
    problem_ids: Tuple[int, ...]
    problems: List[GSMItem]
    target_problem: GSMItem

    # Combine phase
    combined_problem: Optional[str] = None
    combine_error: Optional[str] = None
    combine_completion_tokens: int = 0

    # Solve phase
    solver_output: Optional[str] = None
    solver_answer_raw: Optional[str] = None  # extracted string form (e.g., "7/3" or "12.5")
    solver_answer_fraction: Optional[Fraction] = None
    expected_answer_raw: Optional[str] = None
    expected_answer_fraction: Optional[Fraction] = None
    solve_completion_tokens: int = 0
    solver_outputs: List[str] = dataclasses.field(default_factory=list)
    second_solve_attempt: bool = False
    alerts: List[str] = dataclasses.field(default_factory=list)

    # Outcome
    success: bool = False
    reason: Optional[str] = None  # why failed, if failed
    elapsed_s: float = 0.0


# ------------------------------ Utilities ------------------------------

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                # tolerate malformed lines; skip
                continue
    return out


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    # Minimal robust append (single-line JSON per record)
    for _ in range(3):
        try:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            return
        except Exception:
            time.sleep(0.1)
    raise RuntimeError(f"Failed to append to {path}")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def ensure_text_parts(content: Any, part_type: str) -> List[Dict[str, str]]:
    if isinstance(content, str):
        return [{"type": part_type, "text": content}]
    # Support legacy chat message structures
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                item_type = item.get("type", part_type)
                if item_type in {"input_text", "output_text"}:
                    parts.append({"type": item_type, "text": item["text"]})
                elif item_type == "text":
                    parts.append({"type": part_type, "text": item["text"]})
        if parts:
            return parts
    return [{"type": part_type, "text": str(content)}]


def messages_to_responses_input(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    formatted: List[Dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "assistant":
            part_type = "output_text"
        else:
            part_type = "input_text"
        formatted.append({"role": role, "content": ensure_text_parts(content, part_type)})
    return formatted


def extract_response_text(resp: Any) -> str:
    if hasattr(resp, "output_text") and resp.output_text:
        return resp.output_text
    texts: List[str] = []
    for item in getattr(resp, "output", []) or []:
        for part in getattr(item, "content", []) or []:
            text = getattr(part, "text", None)
            if text:
                texts.append(text)
    if texts:
        return "".join(texts)
    # Fallback for dictionary-like responses
    if isinstance(resp, dict):
        output = resp.get("output") or []
        for item in output:
            for part in item.get("content", []):
                text = part.get("text")
                if text:
                    texts.append(text)
        if texts:
            return "".join(texts)
    return ""


def suppress_third_party_logs() -> None:
    # Keep the console clean for the Rich dashboard.
    for name in [
        "urllib3",
        "openai",
        "httpx",
        "datasets",
        "filelock",
        "fsspec",
        "huggingface_hub",
    ]:
        logging.getLogger(name).setLevel(logging.WARNING)
    if hf_set_verbosity_error is not None:
        try:
            hf_set_verbosity_error()
        except Exception:
            pass


def sha256_text(s: str) -> str:
    return sha256(s.encode("utf-8")).hexdigest()


def parse_fraction_from_string(s: str) -> Optional[Fraction]:
    """
    Parse a numeric string into a Fraction.
    Supports:
      - Integers: "123", "-7"
      - Decimals: "12.5", "1,234.56" (commas stripped)
      - Fractions: "7/3", "-10/4"
      - Mixed numbers: "1 1/2"
      - Percent: "45%" -> 45 (we treat as the number 45, not 0.45, to align with GSM8K conventions)
      - Currency signs and spaces are ignored.
    Returns None if parsing fails.
    """
    if s is None:
        return None
    t = s.strip()
    if not t:
        return None
    # Remove currency symbols and surrounding noise
    t = t.replace("$", "").replace("£", "").replace("€", "")
    t = t.replace(",", "")
    t = t.strip()

    # If it looks like a mixed number "a b/c"
    m = re.fullmatch(r"(-?\d+)\s+(\d+)\s*/\s*(\d+)", t)
    if m:
        whole = int(m.group(1))
        num = int(m.group(2))
        den = int(m.group(3))
        if den == 0:
            return None
        frac = Fraction(num, den)
        return Fraction(whole, 1) + (frac if whole >= 0 else -frac)

    # If it looks like a plain fraction "a/b"
    m = re.fullmatch(r"(-?\d+)\s*/\s*(-?\d+)", t)
    if m:
        num = int(m.group(1))
        den = int(m.group(2))
        if den == 0:
            return None
        return Fraction(num, den)

    # If it ends with percent, strip the percent sign and parse the number as-is
    # (So "45%" -> 45, consistent with many GSM8K answers.)
    if t.endswith("%"):
        t = t[:-1].strip()

    # If it's a decimal or integer
    # Use Decimal for exact conversion into Fraction when possible.
    from decimal import Decimal, InvalidOperation, getcontext
    getcontext().prec = 50
    try:
        d = Decimal(t)
        # Convert Decimal to Fraction exactly
        return Fraction(d)
    except InvalidOperation:
        return None


def extract_last_numeric_token(text: str) -> Optional[str]:
    """
    Extract a "best guess" numeric token from arbitrary text.
    We prefer the last numeric-like pattern (often the final answer).
    Supports integers, decimals, fractions, mixed numbers, and percentages.
    """
    if text is None:
        return None
    # Candidate patterns (order matters: longer/more specific first)
    patterns = [
        r"-?\d+\s+\d+\s*/\s*\d+",       # mixed number "1 1/2"
        r"-?\d+\s*/\s*-?\d+",          # fraction "7/3"
        r"-?\d+(?:,\d{3})*(?:\.\d+)?%",# percent with decimals
        r"-?\d+(?:,\d{3})*(?:\.\d+)?", # integer or decimal with commas
    ]
    for pat in patterns:
        matches = re.findall(pat, text)
        if matches:
            return matches[-1].strip()
    return None


def extract_gsm8k_final_answer_str(answer_field: str) -> Optional[str]:
    """
    GSM8K convention: final numeric answer is after a '####' marker at the end.
    Example: ".... Therefore, the answer is \\n\\n#### 24"
    This function returns the raw string after '#### ' (not normalized).
    If not found, we fall back to extracting the last numeric token in the whole string.
    """
    if answer_field is None:
        return None
    # Look for '####' marker
    m = re.search(r"####\s*([^\n\r]+)", answer_field)
    if m:
        cand = m.group(1).strip()
        # Keep only the first token on that line
        # (If model included extra commentary after ####, we ignore it.)
        cand = cand.split()[0]
        return cand
    # Fallback: best-effort extraction
    return extract_last_numeric_token(answer_field)


def canonicalize_answer_str(s: str) -> Optional[str]:
    """
    Normalize a numeric string into a minimal canonical string for logging & consistency.
    We prefer irreducible improper fractions; decimals are kept as given if exact.
    """
    frac = parse_fraction_from_string(s)
    if frac is None:
        return None
    # Prefer integer if denominator==1, else fraction "num/den"
    if frac.denominator == 1:
        return str(frac.numerator)
    return f"{frac.numerator}/{frac.denominator}"


def answers_match(a: Fraction, b: Fraction) -> bool:
    """
    Exact rational comparison. If both are Fractions, do exact match.
    """
    return a == b


# ------------------------------ I/O: GSM8K ------------------------------

def download_or_load_gsm8k_split(
    dst_in_dataset_dir: Path,
    console: Optional[Console],
    split: str,
) -> List[GSMItem]:
    """
    Load a GSM8K split from Hugging Face datasets.
    If already materialized as JSONL under in_dataset/gsm8k_<split>.jsonl, load from there.
    Otherwise, download via `datasets`, assign sequential IDs, and save.
    Returns the in-memory list of GSMItem with parsed expected answers.
    """
    ensure_dir(dst_in_dataset_dir)
    jsonl_path = dst_in_dataset_dir / f"gsm8k_{split}.jsonl"

    items: List[GSMItem] = []

    if jsonl_path.exists():
        # Read pre-materialized
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                # Backward compat tolerance
                q = obj.get("question")
                a = obj.get("answer")
                i = obj.get("id")
                exp_raw = extract_gsm8k_final_answer_str(a)
                exp_frac = parse_fraction_from_string(exp_raw) if exp_raw else None
                items.append(GSMItem(id=i, question=q, answer=a,
                                     expected_final_raw=exp_raw,
                                     expected_final_fraction=exp_frac))
        if console:
            console.log(
                f"[green]Loaded cached GSM8K {split}[/] from {jsonl_path} ({len(items)} items)."
            )
        return items

    # Need to download
    if load_dataset is None:
        raise RuntimeError(
            "The `datasets` package is required but not installed. "
            "Install with: pip install datasets"
        )

    if console:
        console.log(
            f"[cyan]Downloading GSM8K ({split} split, 'main' config) from Hugging Face...[/]"
        )

    try:
        ds = load_dataset("gsm8k", "main", split=split)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load GSM8K via `datasets`: {e}.\n"
            "Ensure network access and `pip install datasets`."
        )

    # Assign sequential IDs in order encountered
    for idx, row in enumerate(ds):
        q = row["question"]
        a = row["answer"]
        exp_raw = extract_gsm8k_final_answer_str(a)
        exp_frac = parse_fraction_from_string(exp_raw) if exp_raw else None
        items.append(GSMItem(id=idx, question=q, answer=a,
                             expected_final_raw=exp_raw,
                             expected_final_fraction=exp_frac))

    # Materialize to JSONL for reproducibility and restartability
    with jsonl_path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps({"id": it.id, "question": it.question, "answer": it.answer},
                               ensure_ascii=False) + "\n")

    if console:
        console.log(f"[green]Saved GSM8K {split}[/] to {jsonl_path} ({len(items)} items).")
    return items


# ------------------------------ Azure OpenAI client ------------------------------

@dataclass
class AzureConfig:
    api_key: str
    endpoint: str
    api_version: str
    deployment: str

    @staticmethod
    def from_env() -> 'AzureConfig':
        key = os.environ.get("AZURE_OPENAI_API_KEY", "").strip()
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip()
        version = os.environ.get("OPENAI_API_VERSION", "").strip()
        deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "").strip()
        missing = [name for name, val in [
            ("AZURE_OPENAI_API_KEY", key),
            ("AZURE_OPENAI_ENDPOINT", endpoint),
            ("OPENAI_API_VERSION", version),
            ("AZURE_OPENAI_DEPLOYMENT", deployment),
        ] if not val]
        if missing:
            raise RuntimeError(
                f"Missing required environment variables for Azure OpenAI: {', '.join(missing)}"
            )
        return AzureConfig(
            api_key=key, endpoint=endpoint, api_version=version, deployment=deployment
        )


class AOAIClient:
    """
    Thin wrapper around Azure OpenAI Chat Completions with retries.
    """

    def __init__(
        self,
        cfg: AzureConfig,
        timeout: float = DEFAULT_REQ_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_base: float = DEFAULT_BACKOFF_BASE,
        backoff_jitter: Tuple[float, float] = DEFAULT_BACKOFF_JITTER,
    ) -> None:
        if AzureOpenAI is None:
            raise RuntimeError(
                "The `openai` package (v1) is required but not installed. "
                "Install with: pip install openai==1.*"
            )
        self.client = AzureOpenAI(
            azure_endpoint=cfg.endpoint,
            api_key=cfg.api_key,
            api_version=cfg.api_version,
            timeout=timeout,
        )
        self.deployment = cfg.deployment
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_jitter = backoff_jitter

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_completion_tokens: Optional[int] = None,
        on_retry: Optional[Callable[[int, Exception], None]] = None,
        reasoning_effort: Optional[str] = None,
    ) -> ChatResult:
        """
        Call Azure OpenAI Responses API with robust retries.
        Returns the content string.
        """
        attempt = 0
        while True:
            try:
                kwargs: Dict[str, Any] = {
                    "model": self.deployment,
                    "input": messages_to_responses_input(messages),
                }
                if temperature is not None:
                    kwargs["temperature"] = temperature
                if max_completion_tokens is not None:
                    kwargs["max_output_tokens"] = max_completion_tokens
                if reasoning_effort is not None:
                    kwargs["reasoning"] = {"effort": reasoning_effort}
                resp = self.client.responses.create(**kwargs)
                content = extract_response_text(resp)
                usage = getattr(resp, "usage", None)
                input_tokens = getattr(usage, "input_tokens", 0) or 0
                output_tokens = getattr(usage, "output_tokens", 0) or 0
                total_tokens = getattr(usage, "total_tokens", 0) or 0
                return ChatResult(
                    content=content.strip(),
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                )
            except Exception as e:
                attempt += 1
                if attempt > self.max_retries:
                    if on_retry is not None:
                        try:
                            on_retry(attempt, e)
                        except Exception:
                            pass
                    raise
                if on_retry is not None:
                    try:
                        on_retry(attempt, e)
                    except Exception:
                        pass
                # Exponential backoff with jitter
                sleep_s = (self.backoff_base ** (attempt - 1)) + random.uniform(*self.backoff_jitter)
                time.sleep(min(60.0, sleep_s))


# ------------------------------ Worker orchestration ------------------------------

@dataclass
class Paths:
    split: str
    root: Path
    in_dataset: Path
    working_dir: Path
    out_dataset: Path
    success_dir: Path
    fail_dir: Path
    success_jsonl: Path
    success_meta: Path
    fail_jsonl: Path
    attempts: Path
    seen_combos: Path
    combo_size: int


def make_paths(root: Path, combo_size: int, split: str) -> Paths:
    base = root / f"gsm8k_{combo_size}probs"
    in_dataset = base / "in_dataset"
    working_dir = base / "working_dir"
    out_dataset = base / "out_dataset"
    success_dir = out_dataset / f"problems{combo_size}_success"
    fail_dir = out_dataset / f"problems{combo_size}_fail"
    ensure_dir(in_dataset)
    ensure_dir(working_dir)
    ensure_dir(success_dir)
    ensure_dir(fail_dir)
    split_suffix = "" if split == "train" else f"_{split}"
    return Paths(
        split=split,
        root=base,
        in_dataset=in_dataset,
        working_dir=working_dir,
        out_dataset=out_dataset,
        success_dir=success_dir,
        fail_dir=fail_dir,
        success_jsonl=success_dir / ("train.jsonl" if split == "train" else f"{split}.jsonl"),
        success_meta=success_dir / ("meta.jsonl" if split == "train" else f"meta{split_suffix}.jsonl"),
        fail_jsonl=fail_dir / ("train.jsonl" if split == "train" else f"{split}.jsonl"),
        attempts=working_dir / ("attempts.jsonl" if split == "train" else f"attempts{split_suffix}.jsonl"),
        seen_combos=working_dir / ("seen_combos.txt" if split == "train" else f"seen_combos{split_suffix}.txt"),
        combo_size=combo_size,
    )


@dataclass
class WorkerStat:
    attempts: int = 0
    successes: int = 0
    failures: int = 0
    retries: int = 0
    completion_tokens: int = 0
    status: str = "Idle"
    last_event: str = ""
    last_duration: float = 0.0
    current_combo: Optional[Tuple[int, ...]] = None
    stage_started_at: float = dataclasses.field(default_factory=time.time)
    first_started_at: float = dataclasses.field(default_factory=time.time)
    last_update: float = dataclasses.field(default_factory=time.time)
    last_error: Optional[str] = None
    log_message: Optional[str] = None

    def stage_elapsed(self) -> float:
        return max(0.0, time.time() - self.stage_started_at)


@dataclass
class SharedState:
    target_success: int
    max_workers: int
    successes: int = 0
    failures: int = 0
    retries: int = 0
    initial_successes: int = 0
    initial_failures: int = 0
    completion_tokens_total: int = 0
    start_time: float = dataclasses.field(default_factory=time.time)
    stop_event: threading.Event = dataclasses.field(default_factory=threading.Event)
    lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)
    io_lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)
    last_event: str = ""
    last_error: Optional[str] = None
    dashboard_event: threading.Event = dataclasses.field(default_factory=threading.Event)
    # Per-worker stats
    worker_stats: Dict[int, WorkerStat] = dataclasses.field(default_factory=dict)


def read_seen_combos(path: Path) -> set:
    seen = set()
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                t = line.strip()
                if not t:
                    continue
                seen.add(t)
    return seen


def append_seen_combo(path: Path, combo: Tuple[int, ...]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(",".join(str(x) for x in combo) + "\n")


def notify_dashboard(state: SharedState) -> None:
    state.dashboard_event.set()


_CURRENT_COMBO_SENTINEL = object()


def record_worker_alert(state: SharedState, wid: int, message: str) -> None:
    with state.lock:
        stat = state.worker_stats.setdefault(wid, WorkerStat())
        stat.log_message = message
    notify_dashboard(state)


def update_worker_status(
    state: SharedState,
    wid: int,
    *,
    status: Optional[str] = None,
    event: Optional[str] = None,
    current_combo: Any = _CURRENT_COMBO_SENTINEL,
    stage_reset: bool = True,
) -> None:
    """Helper to mutate per-worker stats under the shared lock."""
    with state.lock:
        stat = state.worker_stats.setdefault(wid, WorkerStat())
        if status is not None:
            stat.status = status
        if event is not None:
            stat.last_event = event
        if current_combo is not _CURRENT_COMBO_SENTINEL:
            stat.current_combo = current_combo
        if stage_reset:
            now = time.time()
            stat.stage_started_at = now
            stat.last_update = now
        else:
            stat.last_update = time.time()
    notify_dashboard(state)


def pick_random_combo(n_items: int, combo_size: int, rng: random.Random) -> Tuple[int, ...]:
    if combo_size > n_items:
        raise ValueError("combo_size cannot exceed number of available items")
    return tuple(rng.sample(range(n_items), combo_size))


def build_combine_prompt(problem_texts: List[str]) -> str:
    problem_blocks = []
    for idx, text in enumerate(problem_texts, start=1):
        block = f"Problem {idx}:\n{text.strip()}"
        problem_blocks.append(block)
    return COMBINE_PROMPT_TEMPLATE.format(
        n=len(problem_texts),
        graph_instructions=GRAPH_INSTRUCTIONS[len(problem_texts)-1].format(n=len(problem_texts)),
        problem_blocks="\n\n".join(problem_blocks),
    )


def do_attempt(
    client: AOAIClient,
    items: List[GSMItem],
    combo: Tuple[int, ...],
    progress_cb: Optional[Callable[[str], None]] = None,
    combine_chat_kwargs: Optional[Dict[str, Any]] = None,
    solve_chat_kwargs: Optional[Dict[str, Any]] = None,
    combine_retry_cb: Optional[Callable[[int, Exception], None]] = None,
    solve_retry_cb: Optional[Callable[[int, Exception], None]] = None,
) -> AttemptResult:
    t0 = time.time()
    problems = [items[i] for i in combo]
    target_problem = problems[-1]
    res = AttemptResult(problem_ids=combo, problems=problems, target_problem=target_problem)
    res.expected_answer_raw = target_problem.expected_final_raw
    res.expected_answer_fraction = target_problem.expected_final_fraction

    # 1) Combine
    combine_user_content = build_combine_prompt([p.question for p in problems])
    messages = [
        {"role": "system", "content": SYSTEM_COMBINE},
        {"role": "user", "content": combine_user_content},
    ]
    if progress_cb:
        progress_cb("combine")

    try:
        chat_kwargs = dict(combine_chat_kwargs or {})
        if combine_retry_cb is not None:
            chat_kwargs.setdefault("on_retry", combine_retry_cb)
        combined_res = client.chat(messages, **chat_kwargs)
        res.combined_problem = combined_res.content
        res.combine_completion_tokens = combined_res.output_tokens
    except Exception as e:
        res.combine_error = f"combine_error: {type(e).__name__}: {e}"
        res.reason = "combine_call_failed"
        res.elapsed_s = time.time() - t0
        return res

    if not res.combined_problem or res.combined_problem.upper().startswith("ERROR:"):
        res.combine_error = res.combined_problem or "empty"
        res.reason = "combine_invalid_output"
        res.elapsed_s = time.time() - t0
        return res

    # 2) Solve
    if progress_cb:
        progress_cb("solve")
    solve_user_content = SOLVE_PROMPT.format(problem=res.combined_problem)
    messages = [
        {"role": "system", "content": SYSTEM_SOLVE},
        {"role": "user", "content": solve_user_content},
    ]
    try:
        solve_kwargs = dict(solve_chat_kwargs or {})
        if solve_retry_cb is not None:
            solve_kwargs.setdefault("on_retry", solve_retry_cb)
        solver_res = client.chat(messages, **solve_kwargs)
        res.solver_output = solver_res.content
        res.solve_completion_tokens += solver_res.output_tokens
        res.solver_outputs.append(solver_res.content)
    except Exception as e:
        res.reason = f"solve_call_failed: {type(e).__name__}: {e}"
        res.elapsed_s = time.time() - t0
        return res

    def parse_solver_output(text: Optional[str]) -> Tuple[Optional[str], Optional[Fraction]]:
        token = extract_last_numeric_token(text or "")
        canonical = canonicalize_answer_str(token) if token else None
        frac = parse_fraction_from_string(canonical) if canonical else None
        return canonical, frac

    res.solver_answer_raw, res.solver_answer_fraction = parse_solver_output(res.solver_output)

    if res.expected_answer_fraction is None:
        res.reason = "expected_answer_not_parsable"
        res.elapsed_s = time.time() - t0
        return res

    def solver_matches() -> bool:
        return (
            res.solver_answer_fraction is not None
            and answers_match(res.solver_answer_fraction, res.expected_answer_fraction)
        )

    need_retry = res.solver_answer_fraction is None or not solver_matches()

    if need_retry:
        if progress_cb:
            progress_cb("solve")
        retry_kwargs = dict(solve_kwargs)
        retry_kwargs["reasoning_effort"] = "high"
        res.second_solve_attempt = True
        res.alerts.append(
            f"Triggered second solve attempt with reasoning_effort=high (combo {res.problem_ids})"
        )
        try:
            retry_res = client.chat(messages, **retry_kwargs)
            res.solver_output = retry_res.content
            res.solve_completion_tokens += retry_res.output_tokens
            res.solver_outputs.append(retry_res.content)
        except Exception as e:
            res.reason = f"solve_call_failed_retry: {type(e).__name__}: {e}"
            res.elapsed_s = time.time() - t0
            return res
        res.solver_answer_raw, res.solver_answer_fraction = parse_solver_output(res.solver_output)

    if res.solver_answer_fraction is None:
        res.reason = "solver_answer_not_parsable"
        res.elapsed_s = time.time() - t0
        return res

    if solver_matches():
        res.success = True
        res.elapsed_s = time.time() - t0
        return res

    res.reason = "answer_mismatch"
    res.elapsed_s = time.time() - t0
    return res


def writer_success(paths: Paths, res: AttemptResult, lock: threading.Lock) -> None:
    """
    Append to success dataset and meta. Thread-safe via external lock.
    """
    assert res.success
    expected_fraction_str = (
        f"{res.expected_answer_fraction.numerator}/{res.expected_answer_fraction.denominator}"
        if res.expected_answer_fraction is not None
        else None
    )
    # GSM8K-style minimal answer: only "#### <number>"
    final_ans_str = canonicalize_answer_str(res.expected_answer_raw) or expected_fraction_str
    if final_ans_str is None:
        final_ans_str = ""
    out_row = {
        "question": res.combined_problem,
        "answer": f"#### {final_ans_str}",
    }
    meta_row = {
        "timestamp": now_iso(),
        "split": paths.split,
        "source_ids": list(res.problem_ids),
        "combined_hash": sha256_text(res.combined_problem or ""),
        "expected_answer_raw": res.expected_answer_raw,
        "expected_answer_fraction": expected_fraction_str,
        "elapsed_s": round(res.elapsed_s, 3),
        "combine_completion_tokens": res.combine_completion_tokens,
        "solve_completion_tokens": res.solve_completion_tokens,
        "second_solve_attempt": res.second_solve_attempt,
        "solver_outputs": res.solver_outputs,
    }
    with lock:
        append_jsonl(paths.success_jsonl, out_row)
        append_jsonl(paths.success_meta, meta_row)
        append_jsonl(paths.attempts, {"success": True, **meta_row})


def writer_fail(paths: Paths, res: AttemptResult, lock: threading.Lock) -> None:
    """
    Append to failure dataset with detailed diagnostics. Thread-safe via external lock.
    """
    solver_fraction_str = (
        f"{res.solver_answer_fraction.numerator}/{res.solver_answer_fraction.denominator}"
        if res.solver_answer_fraction is not None
        else None
    )
    expected_fraction_str = (
        f"{res.expected_answer_fraction.numerator}/{res.expected_answer_fraction.denominator}"
        if res.expected_answer_fraction is not None
        else None
    )
    problems_payload = [
        {
            "sequence": idx + 1,
            "id": problem.id,
            "question": problem.question,
            "answer": problem.answer,
        }
        for idx, problem in enumerate(res.problems)
    ]
    row = {
        "timestamp": now_iso(),
        "split": paths.split,
        "source_ids": list(res.problem_ids),
        "problems": problems_payload,
        "target_problem_sequence": len(res.problems),
        "combined_problem": res.combined_problem,
        "combine_error": res.combine_error,
        "solver_output": res.solver_output,
        "solver_answer_raw": res.solver_answer_raw,
        "expected_answer_raw": res.expected_answer_raw,
        "solver_answer_fraction": solver_fraction_str,
        "expected_answer_fraction": expected_fraction_str,
        "reason": res.reason,
        "elapsed_s": round(res.elapsed_s, 3),
        "combine_completion_tokens": res.combine_completion_tokens,
        "solve_completion_tokens": res.solve_completion_tokens,
        "second_solve_attempt": res.second_solve_attempt,
        "solver_outputs": res.solver_outputs,
    }
    with lock:
        append_jsonl(paths.fail_jsonl, row)
        append_jsonl(paths.attempts, {"success": False, **row})


def summarize_failure(res: AttemptResult) -> str:
    reason = res.reason or "unknown_failure"
    parts: List[str] = []
    if res.combine_error:
        parts.append(res.combine_error)
    if res.reason in {"solve_call_failed", "solve_call_failed_retry", "solver_answer_not_parsable", "answer_mismatch"}:
        if res.solver_output:
            parts.append(f"solver_output={res.solver_output}")
    if res.reason == "answer_mismatch":
        parts.append(f"expected={res.expected_answer_raw}")
        parts.append(f"got={res.solver_answer_raw}")
    if res.reason == "solver_answer_not_parsable" and res.solver_answer_raw:
        parts.append(f"parsed={res.solver_answer_raw}")
    if not parts and res.solver_output and res.reason not in {"combine_call_failed", "combine_invalid_output"}:
        parts.append(f"solver_output={res.solver_output}")
    if res.reason == "expected_answer_not_parsable" and res.expected_answer_raw:
        parts.append(f"expected={res.expected_answer_raw}")
    if res.reason == "combine_invalid_output" and res.combined_problem:
        parts.append(f"combined_problem={res.combined_problem}")
    if res.second_solve_attempt:
        parts.append("second_solve_attempt=high")
    if res.solver_outputs:
        parts.append(f"solver_outputs={res.solver_outputs}")
    detail = "; ".join(filter(None, parts))
    return f"{reason}{(': ' + detail) if detail else ''}"


def worker_loop(
    wid: int,
    state: SharedState,
    paths: Paths,
    client: AOAIClient,
    items: List[GSMItem],
    seen: set,
    rng: random.Random,
    combine_chat_kwargs: Optional[Dict[str, Any]],
    solve_chat_kwargs: Optional[Dict[str, Any]],
) -> None:
    update_worker_status(state, wid, status="Idle", event="worker starting", current_combo=None)
    n = len(items)
    combo_size = paths.combo_size

    while not state.stop_event.is_set():
        with state.lock:
            if state.successes >= state.target_success:
                break

        update_worker_status(state, wid, status="Sampling", event="choosing combo", current_combo=None)

        combo: Optional[Tuple[int, ...]] = None
        for _ in range(1000):
            if state.stop_event.is_set():
                break
            candidate = pick_random_combo(n, combo_size, rng)
            combo_key = ",".join(str(x) for x in candidate)
            with state.lock:
                if combo_key not in seen:
                    seen.add(combo_key)
                    append_seen_combo(paths.seen_combos, candidate)
                    combo = candidate
                    break
        if combo is None:
            if state.stop_event.is_set():
                break
            time.sleep(0.1)
            continue

        update_worker_status(
            state,
            wid,
            status="Queued",
            event=f"combo {combo}",
            current_combo=combo,
        )

        def stage_cb(stage: str) -> None:
            if stage == "combine":
                update_worker_status(
                    state,
                    wid,
                    status="Combining",
                    event=f"combining {combo}",
                    current_combo=combo,
                )
            elif stage == "solve":
                update_worker_status(
                    state,
                    wid,
                    status="Solving",
                    event=f"solving {combo}",
                    current_combo=combo,
                )
            elif stage == "validate":
                update_worker_status(
                    state,
                    wid,
                    status="Validating",
                    event=f"checking {combo}",
                    current_combo=combo,
                )

        def make_retry_cb(stage_label: str) -> Callable[[int, Exception], None]:
            def _cb(attempt_no: int, error: Exception) -> None:
                message = (
                    f"worker {wid}: {stage_label} retry attempt {attempt_no} due to {type(error).__name__}: {error}"
                )
                with state.lock:
                    state.retries += 1
                    state.last_event = message
                    stat = state.worker_stats.setdefault(wid, WorkerStat())
                    stat.retries += 1
                    stat.last_event = message
                    stat.status = f"{stage_label} retry"
                    stat.last_update = time.time()
                    stat.current_combo = combo
                    stat.log_message = message
                notify_dashboard(state)
            return _cb

        attempt_started = time.time()
        try:
            res = do_attempt(
                client,
                items,
                combo,
                progress_cb=stage_cb,
                combine_chat_kwargs=combine_chat_kwargs,
                solve_chat_kwargs=solve_chat_kwargs,
                combine_retry_cb=make_retry_cb("Combining"),
                solve_retry_cb=make_retry_cb("Solving"),
            )
        except Exception as e:
            combo_problems = [items[i] for i in combo]
            res = AttemptResult(
                problem_ids=combo,
                problems=combo_problems,
                target_problem=combo_problems[-1],
                success=False,
                reason=f"unexpected_exception: {type(e).__name__}: {e}",
            )
            res.expected_answer_raw = combo_problems[-1].expected_final_raw
            res.expected_answer_fraction = combo_problems[-1].expected_final_fraction
            res.elapsed_s = time.time() - attempt_started

        if res.elapsed_s == 0:
            res.elapsed_s = time.time() - attempt_started

        stage_cb("validate")

        attempt_tokens = res.combine_completion_tokens + res.solve_completion_tokens

        if res.alerts:
            record_worker_alert(state, wid, res.alerts[-1])

        if res.success:
            with state.lock:
                state.successes += 1
                last_event = (
                    f"worker {wid}: success #{state.successes} in {res.elapsed_s:.2f}s "
                    f"(combo {res.problem_ids})"
                )
                state.last_event = last_event
                state.completion_tokens_total += attempt_tokens
                stat = state.worker_stats.setdefault(wid, WorkerStat())
                stat.successes += 1
                stat.attempts += 1
                stat.last_duration = res.elapsed_s
                stat.last_event = last_event
                stat.status = "Success"
                stat.last_update = time.time()
                stat.stage_started_at = stat.last_update
                stat.current_combo = None
                stat.completion_tokens += attempt_tokens
            writer_success(paths, res, state.io_lock)
            notify_dashboard(state)
        else:
            failure_detail = summarize_failure(res)
            with state.lock:
                state.failures += 1
                last_event = (
                    f"worker {wid}: fail {failure_detail} (combo {res.problem_ids})"
                )
                state.last_event = last_event
                state.last_error = failure_detail
                state.completion_tokens_total += attempt_tokens
                stat = state.worker_stats.setdefault(wid, WorkerStat())
                stat.failures += 1
                stat.attempts += 1
                stat.last_duration = res.elapsed_s
                stat.last_event = last_event
                stat.status = "Failed"
                stat.last_update = time.time()
                stat.stage_started_at = stat.last_update
                stat.current_combo = None
                stat.last_error = failure_detail
                stat.completion_tokens += attempt_tokens
                stat.log_message = failure_detail
            writer_fail(paths, res, state.io_lock)
            notify_dashboard(state)

        if state.stop_event.is_set():
            break
        with state.lock:
            if state.successes >= state.target_success:
                state.stop_event.set()
                notify_dashboard(state)
                break

    update_worker_status(state, wid, status="Stopped", event="worker exit", current_combo=None)


def format_eta(seconds_left: float) -> str:
    if seconds_left <= 0 or math.isinf(seconds_left) or math.isnan(seconds_left):
        return "-"
    m, s = divmod(int(seconds_left), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def build_dashboard_renderable(state: SharedState) -> Group:
    with state.lock:
        successes = state.successes
        failures = state.failures
        retries = state.retries
        start_time = state.start_time
        target_success = state.target_success
        last_event = state.last_event
        last_error = state.last_error
        stopping = state.stop_event.is_set()
        completion_tokens_total = state.completion_tokens_total
        initial_successes = state.initial_successes
        initial_failures = state.initial_failures
        worker_snapshot = {
            wid: dataclasses.replace(stat)
            for wid, stat in state.worker_stats.items()
        }

    table = Table(box=SIMPLE, show_lines=False, expand=True)
    table.add_column("Role", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Successes", no_wrap=True, justify="right")
    table.add_column("Failures", no_wrap=True, justify="right")
    table.add_column("Attempts", no_wrap=True, justify="right")
    table.add_column("Retries", no_wrap=True, justify="right")
    table.add_column("Success Rate", no_wrap=True, justify="right")
    table.add_column("ETA", no_wrap=True, justify="right")
    table.add_column("Completion Tokens", no_wrap=True, justify="right")
    table.add_column("Stage Seconds", no_wrap=True, justify="right")
    table.add_column("Last Event", overflow="fold")

    elapsed = max(1e-9, time.time() - start_time)
    display_successes = max(0, successes - initial_successes)
    display_failures = max(0, failures - initial_failures)
    display_attempts = display_successes + display_failures
    rate = display_successes / elapsed
    remaining = max(0, target_success - successes)
    eta = remaining / rate if rate > 0 else float("inf")
    master_status = "Stopping" if stopping else ("Done" if remaining == 0 else "Running")
    table.add_row(
        "MASTER",
        master_status,
        str(display_successes),
        str(display_failures),
        str(display_attempts),
        str(retries),
        f"{rate:.2f}",
        format_eta(eta),
        str(completion_tokens_total),
        "-",
        last_event,
    )

    now = time.time()
    for wid in sorted(worker_snapshot.keys()):
        stats = worker_snapshot[wid]
        worker_elapsed = max(1e-9, now - stats.first_started_at)
        per_s = stats.successes / worker_elapsed
        table.add_row(
            f"WORKER-{wid}",
            stats.status,
            str(stats.successes),
            str(stats.failures),
            str(stats.attempts),
            str(stats.retries),
            f"{per_s:.2f}",
            "-",
            str(stats.completion_tokens),
            f"{stats.stage_elapsed():.1f}",
            stats.last_event,
        )

    errors_table = Table(box=SIMPLE, show_header=True, expand=True)
    errors_table.add_column("Role", no_wrap=True)
    errors_table.add_column("Last Error", overflow="fold")
    errors_table.add_row("MASTER", last_error or "-")
    for wid in sorted(worker_snapshot.keys()):
        stats = worker_snapshot[wid]
        errors_table.add_row(f"WORKER-{wid}", stats.last_error or "-")

    alerts_table = Table(box=SIMPLE, show_header=True, expand=True)
    alerts_table.add_column("Worker", no_wrap=True)
    alerts_table.add_column("Latest Alert", overflow="fold")
    for wid in sorted(worker_snapshot.keys()):
        stats = worker_snapshot[wid]
        alert = stats.log_message or "-"
        alerts_table.add_row(f"WORKER-{wid}", alert)

    return Group(
        table,
        Panel(errors_table, title="Last errors", box=SIMPLE),
        Panel(alerts_table, title="Worker alerts", box=SIMPLE),
    )


def live_dashboard(state: SharedState, console: Console) -> None:
    state.dashboard_event.set()
    with Live(build_dashboard_renderable(state), console=console, transient=False) as live:
        state.dashboard_event.clear()
        while not state.stop_event.is_set():
            triggered = state.dashboard_event.wait(timeout=1.0)
            if not triggered and not state.dashboard_event.is_set():
                continue
            state.dashboard_event.clear()
            live.update(build_dashboard_renderable(state))
        # Final refresh before exit
        live.update(build_dashboard_renderable(state))


# ------------------------------ Main ------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Combine GSM8K problems in parallel via Azure OpenAI.")
    p.add_argument("--out_dir", type=str, default=os.environ.get("OUT_DIR", "."),
                   help="Root directory for outputs (default: $OUT_DIR or current directory).")
    p.add_argument("--max_samples", type=int, default=DEFAULT_MAX_SAMPLES,
                   help="Number of successful combined problems to generate (default: 200).")
    p.add_argument("--max_workers", type=int, default=DEFAULT_MAX_WORKERS,
                   help=f"Number of parallel workers (default: {DEFAULT_MAX_WORKERS}).")
    p.add_argument("--max_retries", type=int, default=DEFAULT_MAX_RETRIES,
                   help=f"Max retries per Azure API call (default: {DEFAULT_MAX_RETRIES}).")
    p.add_argument("--timeout", type=float, default=DEFAULT_REQ_TIMEOUT,
                   help=f"Azure API request timeout in seconds (default: {DEFAULT_REQ_TIMEOUT}).")
    p.add_argument("--n_problems", type=int, default=DEFAULT_N_PROBLEMS,
                   help=f"Number of GSM8K problems to combine per attempt (default: {DEFAULT_N_PROBLEMS}).")
    p.add_argument("--split", choices=("train", "test"), default="train",
                   help="GSM8K split to use as the source problems (default: train).")
    p.add_argument("--quiet", action="store_true", help="Reduce console output (hides the live dashboard).")
    p.add_argument("--combine-temperature", type=float, default=None,
                   help="Temperature to use for the combine request (default: provider default).")
    p.add_argument("--combine-max-completion-tokens", type=int, default=None,
                   help="Max completion tokens for the combine request (default: provider default).")
    p.add_argument("--solve-temperature", type=float, default=None,
                   help="Temperature to use for the solve request (default: provider default).")
    p.add_argument("--solve-max-completion-tokens", type=int, default=None,
                   help="Max completion tokens for the solve request (default: provider default).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Console (rich)
    if Console is None or Live is None:
        raise RuntimeError("The `rich` package is required. Install with: pip install rich")
    console = Console()

    combo_size = int(args.n_problems)
    if combo_size < 2:
        console.print("[red]--n_problems must be at least 2 so there is something to combine.[/]")
        sys.exit(2)

    # Logging
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    suppress_third_party_logs()

    # Paths
    out_root = Path(args.out_dir).expanduser().resolve()
    paths = make_paths(out_root, combo_size, args.split)

    # Chat parameter overrides
    combine_chat_kwargs: Optional[Dict[str, Any]] = {}
    if args.combine_temperature is not None:
        combine_chat_kwargs["temperature"] = args.combine_temperature
    if args.combine_max_completion_tokens is not None:
        combine_chat_kwargs["max_completion_tokens"] = args.combine_max_completion_tokens
    if not combine_chat_kwargs:
        combine_chat_kwargs = None

    solve_chat_kwargs: Optional[Dict[str, Any]] = {}
    if args.solve_temperature is not None:
        solve_chat_kwargs["temperature"] = args.solve_temperature
    if args.solve_max_completion_tokens is not None:
        solve_chat_kwargs["max_completion_tokens"] = args.solve_max_completion_tokens
    if not solve_chat_kwargs:
        solve_chat_kwargs = None

    # Azure config
    try:
        aoai_cfg = AzureConfig.from_env()
    except Exception as e:
        console.print(f"[red]Azure OpenAI configuration error:[/] {e}")
        sys.exit(2)

    # AOAI client
    client = AOAIClient(
        aoai_cfg,
        timeout=float(args.timeout),
        max_retries=int(args.max_retries),
    )

    # Ctrl-C handling for graceful stop
    state = SharedState(target_success=int(args.max_samples), max_workers=int(args.max_workers))

    def handle_sigint(signum, frame):
        if not state.stop_event.is_set():
            state.stop_event.set()
            console.log("[yellow]SIGINT received, signaling workers to stop...[/]")
            notify_dashboard(state)
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, handle_sigint)

    # Repeatable randomness (optional seed)
    # Derive one RNG per worker so concurrent sampling stays thread-safe.
    if RNG_SEED is not None:
        seed_source = random.Random(RNG_SEED)
    else:
        seed_source = random.Random()
    worker_rngs = [random.Random(seed_source.randrange(2**63)) for _ in range(state.max_workers)]

    # Load or download GSM8K train
    items = download_or_load_gsm8k_split(paths.in_dataset, console, args.split)
    if combo_size > len(items):
        console.print(
            f"[red]--n_problems of {combo_size} exceeds available GSM8K items ({len(items)}).[/]"
        )
        sys.exit(2)

    # Count existing successes (resumable)
    existing_success = read_jsonl(paths.success_jsonl)
    state.successes = len(existing_success)
    state.initial_successes = state.successes
    existing_failures = count_jsonl(paths.fail_jsonl)
    state.failures = existing_failures
    state.initial_failures = existing_failures
    console.log(f"[green]Resuming[/]: found {state.successes} existing successes. Target: {state.target_success}.")

    # Initialize seen combinations from prior runs
    seen = read_seen_combos(paths.seen_combos)

    with state.lock:
        for wid in range(state.max_workers):
            state.worker_stats[wid] = WorkerStat(status="Idle", last_event="awaiting tasks")
    notify_dashboard(state)

    # Start workers
    threads: List[threading.Thread] = []
    for wid, worker_rng in enumerate(worker_rngs):
        t = threading.Thread(
            target=worker_loop,
            args=(wid, state, paths, client, items, seen, worker_rng, combine_chat_kwargs, solve_chat_kwargs),
            name=f"worker-{wid}",
            daemon=True,
        )
        t.start()
        threads.append(t)

    skip_join = False
    try:
        if not args.quiet:
            live_dashboard(state, console)
        else:
            while not state.stop_event.is_set() and any(t.is_alive() for t in threads):
                time.sleep(0.25)
    except KeyboardInterrupt:
        skip_join = True
        state.stop_event.set()
        console.log("[yellow]Interrupted by user; exiting without waiting for all workers.[/]")

    if not skip_join:
        for t in threads:
            t.join()

    # Final summary
    console.print("\n[bold]Run complete[/].")
    console.print(f"Successes: {state.successes} / Target: {state.target_success}")
    console.print(f"Failures:  {state.failures}")
    console.print(f"Outputs:")
    console.print(f"  - Success dataset: [link=file://{paths.success_jsonl}] {paths.success_jsonl}")
    console.print(f"  - Success meta:    [link=file://{paths.success_meta}] {paths.success_meta}")
    console.print(f"  - Fail dataset:    [link=file://{paths.fail_jsonl}] {paths.fail_jsonl}")


if __name__ == "__main__":
    main()
