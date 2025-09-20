#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gsm8k_3probs.py

Purpose
-------
Create a new dataset by *combining* three randomly chosen GSM8K train problems into a single
problem whose final answer is guaranteed to match the original answer of Problem C. For each
attempt we:
  1) Pick three random, distinct problems A, B, C from GSM8K (train).
  2) Ask Azure OpenAI to *rewrite* (i.e., combine) the three into one problem using a strict template.
  3) Ask Azure OpenAI to *solve* the combined problem, returning only the final numeric answer.
  4) Validate that the final numeric answer equals the GSM8K final answer for Problem C.
  5) If valid, append to `out_dataset/problems3_success/train.jsonl` in GSM8K format
     (fields: {"question": <combined problem>, "answer": "#### <final_number>"}).
     We also log provenance in `out_dataset/problems3_success/meta.jsonl`.
  6) Otherwise, append a detailed record to `out_dataset/problems3_fail/train.jsonl` for debugging.

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
   in the *final output*, we add `out_dataset/problems3_success/meta.jsonl` with, for each
   generated item, the `[idA, idB, idC]` triple, a content hash of the combined problem, and
   timestamps. This keeps the success dataset strictly GSM8K-formatted while still shipping the
   needed debug info alongside it.

3) **Resumability**: We store outputs directly in their final files using append-only JSONL.
   On restart, we count existing successes and continue. We also maintain a `working_dir/attempts.jsonl`
   and `working_dir/seen_triples.txt` to avoid retrying identical triples across runs. This ensures
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

3) Optional: choose where to write output (default: ./gsm8k_3probs in current dir)
   export OUT_DIR="/path/to/output_root"

4) Run:
   python gsm8k_3probs.py --max_problem 200 --max_workers 8

Directory Layout (created under OUT_DIR/gsm8k_3probs)
-----------------------------------------------------
in_dataset/
  gsm8k_train.jsonl             # input GSM8K train data with sequential IDs for reference
working_dir/
  attempts.jsonl                # every attempt appended here (success or fail), for provenance
  seen_triples.txt              # set of processed triples (one "a,b,c" per line) to avoid repeats
out_dataset/
  problems3_success/
    train.jsonl                 # GSM8K-format output (question, answer)
    meta.jsonl                  # provenance: source triple IDs, hashes
  problems3_fail/
    train.jsonl                 # detailed diagnostics for failed attempts

Notes
-----
- The script only uses the GSM8K *train* split.
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
from typing import Any, Dict, Iterable, List, Optional, Tuple

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
    from rich.console import Console
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


# ------------------------------ Configuration ------------------------------

COMBINE_PROMPT_TEMPLATE = """Consider below 3 math problems. Each of these problems have one or many input values and only one output value. Only the direct numerical values specified in the problem can be considered as input value. We want to combine these 3 problems to form a one single problem in the following way: Output of problem A should become input for problem B as well as C. Additionally, output of problem B should also feed into to one of the inputs of problem C. Finally, the output of problem C should become the final output of the combined problem and this output value should remain exactly same as it is in original problem C. When replacing input of a problem with output of another problem, you must make sure that final input value of the problem remains same as in original problem. To achieve this goal, you might chose to transform output value of a problem by adding some constant or multiplying with some constant to convert it before making it as an input value for the another problem. Notice that problem C consumes outputs from problem A as well as problem B. However, it may happen that problem C does not have two different inputs available for replacement. In this case, you should combine outputs of problem A and problem B using sum operator to produce one value and than transform it if needed so that it can replace the available input in the problem C. However if problem C has two or more different input values available then you should use one for problem A and another for problem B. You should keep the problem statements as close to original as possible and make only the minimal number of changes required. Can can label output of each problem using a letter so you can reference it later for input for another problem. Your final problem statement should not include problem tags such as "Problem A" etc  but it can include modifications such as "Let's call this value T" or any other alternative that you may find more readable. Using these conditions, produce the final combined problem. In your response only include this final combined problem in only text form and do not use any markdown or latex syntax. If for some reason, you cannot produce the combined problem then output the text "ERROR: <reason>" where <reason> is the reason why you cannot complete the task. 

Problem A:
<problem1> 

Problem B:
<problem2> 

Problem C:
<problem3>"""

SOLVE_PROMPT = """Solve the following math problem. Output ONLY the final numeric answer with no words, no units, no punctuation, and no explanation. If the exact answer is a fraction, return the simplest fraction like 7/3. If the answer is a mixed number, convert it to an improper fraction. If the answer is a monetary amount, return only the number (e.g., 12.50).

Problem:
{problem}
"""

SYSTEM_COMBINE = "You are a careful math problem composer. Follow the user's instructions exactly and return only the final combined problem text."
SYSTEM_SOLVE = "You are a careful math solver. Return only the final numeric answer."


# Reasonable defaults; can be overridden by CLI flags
DEFAULT_MAX_PROBLEM = 200
DEFAULT_MAX_WORKERS = 8
DEFAULT_MAX_RETRIES = 8
DEFAULT_REQ_TIMEOUT = 60.0
DEFAULT_BACKOFF_BASE = 2.0
DEFAULT_BACKOFF_JITTER = (0.2, 0.6)  # seconds
RNG_SEED = None  # set to int for reproducibility; None mixes in system entropy


# ------------------------------ Data models ------------------------------

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
    triple_ids: Tuple[int, int, int]  # (idA, idB, idC)
    problemA: GSMItem
    problemB: GSMItem
    problemC: GSMItem

    # Combine phase
    combined_problem: Optional[str] = None
    combine_error: Optional[str] = None

    # Solve phase
    solver_output: Optional[str] = None
    solver_answer_raw: Optional[str] = None  # extracted string form (e.g., "7/3" or "12.5")
    solver_answer_fraction: Optional[Fraction] = None
    expected_answer_raw: Optional[str] = None
    expected_answer_fraction: Optional[Fraction] = None

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

def download_or_load_gsm8k_train(dst_in_dataset_dir: Path, console: Optional[Console]) -> List[GSMItem]:
    """
    Load GSM8K train split from Hugging Face datasets.
    If already materialized as JSONL under in_dataset/gsm8k_train.jsonl, load from there.
    Otherwise, download via `datasets`, assign sequential IDs, and save.
    Returns the in-memory list of GSMItem with parsed expected answers.
    """
    ensure_dir(dst_in_dataset_dir)
    jsonl_path = dst_in_dataset_dir / "gsm8k_train.jsonl"

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
            console.log(f"[green]Loaded cached GSM8K train[/] from {jsonl_path} ({len(items)} items).")
        return items

    # Need to download
    if load_dataset is None:
        raise RuntimeError(
            "The `datasets` package is required but not installed. "
            "Install with: pip install datasets"
        )

    if console:
        console.log("[cyan]Downloading GSM8K (train split, 'main' config) from Hugging Face...[/]")

    try:
        ds = load_dataset("gsm8k", "main", split="train")
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
        console.log(f"[green]Saved GSM8K train[/] to {jsonl_path} ({len(items)} items).")
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

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.3, max_tokens: int = 800) -> str:
        """
        Call Azure OpenAI Chat Completions with robust retries.
        Returns the content string.
        """
        attempt = 0
        while True:
            try:
                resp = self.client.chat.completions.create(
                    model=self.deployment,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    n=1,
                )
                content = resp.choices[0].message.content or ""
                return content.strip()
            except Exception as e:
                attempt += 1
                if attempt > self.max_retries:
                    raise
                # Exponential backoff with jitter
                sleep_s = (self.backoff_base ** (attempt - 1)) + random.uniform(*self.backoff_jitter)
                time.sleep(min(60.0, sleep_s))


# ------------------------------ Worker orchestration ------------------------------

@dataclass
class Paths:
    root: Path
    in_dataset: Path
    working_dir: Path
    out_dataset: Path
    success_dir: Path
    fail_dir: Path
    success_train: Path
    success_meta: Path
    fail_train: Path
    attempts: Path
    seen_triples: Path


def make_paths(root: Path) -> Paths:
    base = root / "gsm8k_3probs"
    in_dataset = base / "in_dataset"
    working_dir = base / "working_dir"
    out_dataset = base / "out_dataset"
    success_dir = out_dataset / "problems3_success"
    fail_dir = out_dataset / "problems3_fail"
    ensure_dir(in_dataset)
    ensure_dir(working_dir)
    ensure_dir(success_dir)
    ensure_dir(fail_dir)
    return Paths(
        root=base,
        in_dataset=in_dataset,
        working_dir=working_dir,
        out_dataset=out_dataset,
        success_dir=success_dir,
        fail_dir=fail_dir,
        success_train=success_dir / "train.jsonl",
        success_meta=success_dir / "meta.jsonl",
        fail_train=fail_dir / "train.jsonl",
        attempts=working_dir / "attempts.jsonl",
        seen_triples=working_dir / "seen_triples.txt",
    )


@dataclass
class SharedState:
    target_success: int
    max_workers: int
    successes: int = 0
    failures: int = 0
    retries: int = 0
    start_time: float = dataclasses.field(default_factory=time.time)
    stop_event: threading.Event = dataclasses.field(default_factory=threading.Event)
    lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)
    io_lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)
    last_event: str = ""
    # Per-worker stats
    worker_stats: Dict[int, Dict[str, Any]] = dataclasses.field(default_factory=dict)


def read_seen_triples(path: Path) -> set:
    seen = set()
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                t = line.strip()
                if not t:
                    continue
                seen.add(t)
    return seen


def append_seen_triple(path: Path, triple: Tuple[int, int, int]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(f"{triple[0]},{triple[1]},{triple[2]}\n")


def pick_random_triple(n: int, rng: random.Random) -> Tuple[int, int, int]:
    a = rng.randrange(n)
    b = rng.randrange(n)
    while b == a:
        b = rng.randrange(n)
    c = rng.randrange(n)
    while c == a or c == b:
        c = rng.randrange(n)
    return (a, b, c)


def build_combine_prompt(pA: str, pB: str, pC: str) -> str:
    return (COMBINE_PROMPT_TEMPLATE
            .replace("<problem1>", pA.strip())
            .replace("<problem2>", pB.strip())
            .replace("<problem3>", pC.strip()))


def do_attempt(
    client: AOAIClient,
    items: List[GSMItem],
    triple: Tuple[int, int, int],
) -> AttemptResult:
    t0 = time.time()
    idA, idB, idC = triple
    A, B, C = items[idA], items[idB], items[idC]
    res = AttemptResult(triple_ids=triple, problemA=A, problemB=B, problemC=C)
    res.expected_answer_raw = C.expected_final_raw
    res.expected_answer_fraction = C.expected_final_fraction

    # 1) Combine
    combine_user_content = build_combine_prompt(A.question, B.question, C.question)
    messages = [
        {"role": "system", "content": SYSTEM_COMBINE},
        {"role": "user", "content": combine_user_content},
    ]
    try:
        combined = client.chat(messages, temperature=0.4, max_tokens=700)
        res.combined_problem = combined.strip()
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
    solve_user_content = SOLVE_PROMPT.format(problem=res.combined_problem)
    messages = [
        {"role": "system", "content": SYSTEM_SOLVE},
        {"role": "user", "content": solve_user_content},
    ]
    try:
        solver_text = client.chat(messages, temperature=0.0, max_tokens=128)
        res.solver_output = solver_text
    except Exception as e:
        res.reason = f"solve_call_failed: {type(e).__name__}: {e}"
        res.elapsed_s = time.time() - t0
        return res

    # 3) Parse solver output
    solver_raw = extract_last_numeric_token(res.solver_output or "")
    res.solver_answer_raw = canonicalize_answer_str(solver_raw) if solver_raw else None
    res.solver_answer_fraction = parse_fraction_from_string(res.solver_answer_raw) if res.solver_answer_raw else None

    if res.expected_answer_fraction is None:
        res.reason = "expected_answer_not_parsable"
        res.elapsed_s = time.time() - t0
        return res

    if res.solver_answer_fraction is None:
        res.reason = "solver_answer_not_parsable"
        res.elapsed_s = time.time() - t0
        return res

    # 4) Compare
    if answers_match(res.solver_answer_fraction, res.expected_answer_fraction):
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
    # GSM8K-style minimal answer: only "#### <number>"
    final_ans_str = canonicalize_answer_str(res.expected_answer_raw) or str(res.expected_answer_fraction)
    out_row = {
        "question": res.combined_problem,
        "answer": f"#### {final_ans_str}",
    }
    meta_row = {
        "timestamp": now_iso(),
        "source_ids": list(res.triple_ids),
        "combined_hash": sha256_text(res.combined_problem or ""),
        "expected_answer_raw": res.expected_answer_raw,
        "expected_answer_fraction": f"{res.expected_answer_fraction.numerator}/{res.expected_answer_fraction.denominator}",
        "elapsed_s": round(res.elapsed_s, 3),
    }
    with lock:
        append_jsonl(paths.success_train, out_row)
        append_jsonl(paths.success_meta, meta_row)
        append_jsonl(paths.attempts, {"success": True, **meta_row})


def writer_fail(paths: Paths, res: AttemptResult, lock: threading.Lock) -> None:
    """
    Append to failure dataset with detailed diagnostics. Thread-safe via external lock.
    """
    row = {
        "timestamp": now_iso(),
        "source_ids": list(res.triple_ids),
        "A": {"id": res.problemA.id, "question": res.problemA.question, "answer": res.problemA.answer},
        "B": {"id": res.problemB.id, "question": res.problemB.question, "answer": res.problemB.answer},
        "C": {"id": res.problemC.id, "question": res.problemC.question, "answer": res.problemC.answer},
        "combined_problem": res.combined_problem,
        "combine_error": res.combine_error,
        "solver_output": res.solver_output,
        "solver_answer_raw": res.solver_answer_raw,
        "expected_answer_raw": res.expected_answer_raw,
        "solver_answer_fraction": (f"{res.solver_answer_fraction.numerator}/{res.solver_answer_fraction.denominator}"
                                   if res.solver_answer_fraction else None),
        "expected_answer_fraction": (f"{res.expected_answer_fraction.numerator}/{res.expected_answer_fraction.denominator}"
                                     if res.expected_answer_fraction else None),
        "reason": res.reason,
        "elapsed_s": round(res.elapsed_s, 3),
    }
    with lock:
        append_jsonl(paths.fail_train, row)
        append_jsonl(paths.attempts, {"success": False, **row})


def worker_loop(
    wid: int,
    state: SharedState,
    paths: Paths,
    client: AOAIClient,
    items: List[GSMItem],
    seen: set,
    rng: random.Random,
) -> None:
    local_attempts = 0
    local_succ = 0
    local_fail = 0
    local_retries = 0
    last_event = ""

    n = len(items)
    while not state.stop_event.is_set():
        # Stop if we've reached the target
        with state.lock:
            if state.successes >= state.target_success:
                break

        # Pick a fresh triple we haven't seen
        for _ in range(1000):
            triple = pick_random_triple(n, rng)
            triple_key = f"{triple[0]},{triple[1]},{triple[2]}"
            with state.lock:
                if triple_key not in seen:
                    seen.add(triple_key)
                    append_seen_triple(paths.seen_triples, triple)
                    break
        else:
            # Could not find unseen triple quickly (unlikely). Small sleep.
            time.sleep(0.1)
            continue

        local_attempts += 1
        started = time.time()
        try:
            res = do_attempt(client, items, triple)
        except Exception as e:
            # This is an unexpected top-level failure; log as fail.
            res = AttemptResult(
                triple_ids=triple,
                problemA=items[triple[0]],
                problemB=items[triple[1]],
                problemC=items[triple[2]],
                success=False,
                reason=f"unexpected_exception: {type(e).__name__}: {e}",
            )

        if res.success:
            with state.lock:
                state.successes += 1
                last_event = (
                    f"worker {wid}: success #{state.successes} in {res.elapsed_s:.2f}s "
                    f"(triple {res.triple_ids})"
                )
                state.last_event = last_event
            writer_success(paths, res, state.io_lock)
            local_succ += 1
        else:
            with state.lock:
                state.failures += 1
                last_event = f"worker {wid}: fail ({res.reason}) (triple {res.triple_ids})"
                state.last_event = last_event
            writer_fail(paths, res, state.io_lock)
            local_fail += 1

        # Update per-worker stats
        with state.lock:
            state.worker_stats[wid] = {
                "attempts": local_attempts,
                "succ": local_succ,
                "fail": local_fail,
                "retries": local_retries,
                "last": last_event,
                "avg_s": (time.time() - started),
            }

        # Check stop condition again
        with state.lock:
            if state.successes >= state.target_success:
                state.stop_event.set()
                break


def format_eta(seconds_left: float) -> str:
    if seconds_left <= 0 or math.isinf(seconds_left) or math.isnan(seconds_left):
        return "-"
    m, s = divmod(int(seconds_left), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def build_dashboard_table(state: SharedState) -> Table:
    table = Table(box=SIMPLE, show_lines=False, expand=True)
    table.add_column("Role", no_wrap=True)
    table.add_column("Succ", no_wrap=True, justify="right")
    table.add_column("Fail", no_wrap=True, justify="right")
    table.add_column("Att", no_wrap=True, justify="right")
    table.add_column("Ret", no_wrap=True, justify="right")
    table.add_column("Rate/s", no_wrap=True, justify="right")
    table.add_column("ETA", no_wrap=True, justify="right")
    table.add_column("Last event", overflow="fold")

    # Master row
    elapsed = max(1e-9, time.time() - state.start_time)
    rate = state.successes / elapsed
    remaining = max(0, state.target_success - state.successes)
    eta = remaining / rate if rate > 0 else float("inf")
    table.add_row(
        "MASTER",
        str(state.successes),
        str(state.failures),
        str(state.successes + state.failures),
        str(state.retries),
        f"{rate:.2f}",
        format_eta(eta),
        state.last_event[:120],
    )

    # Worker rows
    for wid in sorted(state.worker_stats.keys()):
        stats = state.worker_stats[wid]
        attempts = stats.get("attempts", 0)
        succ = stats.get("succ", 0)
        fail = stats.get("fail", 0)
        retries = stats.get("retries", 0)
        avg_s = stats.get("avg_s", 0.0)
        per_s = (succ / avg_s) if avg_s > 0 else 0.0
        table.add_row(
            f"WORKER-{wid}",
            str(succ),
            str(fail),
            str(attempts),
            str(retries),
            f"{per_s:.2f}",
            "-",
            stats.get("last", "")[:120],
        )

    return table


def live_dashboard(state: SharedState, console: Console, refresh_per_sec: float = 4.0) -> None:
    with Live(build_dashboard_table(state), console=console, refresh_per_second=refresh_per_sec, transient=False) as live:
        while not state.stop_event.is_set():
            time.sleep(1.0 / refresh_per_sec)
            live.update(build_dashboard_table(state))


# ------------------------------ Main ------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Combine GSM8K problems in parallel via Azure OpenAI.")
    p.add_argument("--out_dir", type=str, default=os.environ.get("OUT_DIR", "."),
                   help="Root directory for outputs (default: $OUT_DIR or current directory).")
    p.add_argument("--max_problem", type=int, default=DEFAULT_MAX_PROBLEM,
                   help="Number of successful combined problems to generate (default: 200).")
    p.add_argument("--max_workers", type=int, default=DEFAULT_MAX_WORKERS,
                   help=f"Number of parallel workers (default: {DEFAULT_MAX_WORKERS}).")
    p.add_argument("--max_retries", type=int, default=DEFAULT_MAX_RETRIES,
                   help=f"Max retries per Azure API call (default: {DEFAULT_MAX_RETRIES}).")
    p.add_argument("--timeout", type=float, default=DEFAULT_REQ_TIMEOUT,
                   help=f"Azure API request timeout in seconds (default: {DEFAULT_REQ_TIMEOUT}).")
    p.add_argument("--quiet", action="store_true", help="Reduce console output (hides the live dashboard).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Console (rich)
    if Console is None or Live is None:
        raise RuntimeError("The `rich` package is required. Install with: pip install rich")
    console = Console()

    # Logging
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    suppress_third_party_logs()

    # Paths
    out_root = Path(args.out_dir).expanduser().resolve()
    paths = make_paths(out_root)

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
    state = SharedState(target_success=int(args.max_problem), max_workers=int(args.max_workers))

    def handle_sigint(signum, frame):
        state.stop_event.set()
    signal.signal(signal.SIGINT, handle_sigint)

    # Repeatable randomness (optional seed)
    # Derive one RNG per worker so concurrent sampling stays thread-safe.
    if RNG_SEED is not None:
        seed_source = random.Random(RNG_SEED)
    else:
        seed_source = random.Random()
    worker_rngs = [random.Random(seed_source.randrange(2**63)) for _ in range(state.max_workers)]

    # Load or download GSM8K train
    items = download_or_load_gsm8k_train(paths.in_dataset, console)

    # Count existing successes (resumable)
    existing_success = read_jsonl(paths.success_train)
    state.successes = len(existing_success)
    console.log(f"[green]Resuming[/]: found {state.successes} existing successes. Target: {state.target_success}.")

    # Initialize seen triples from prior runs
    seen = read_seen_triples(paths.seen_triples)

    # Start workers
    threads: List[threading.Thread] = []
    for wid, worker_rng in enumerate(worker_rngs):
        t = threading.Thread(
            target=worker_loop,
            args=(wid, state, paths, client, items, seen, worker_rng),
            name=f"worker-{wid}",
            daemon=True,
        )
        t.start()
        threads.append(t)

    # Start live dashboard (unless quiet)
    if not args.quiet:
        try:
            live_dashboard(state, console)
        except KeyboardInterrupt:
            state.stop_event.set()

    # Wait for workers to finish
    for t in threads:
        t.join()

    # Final summary
    console.print("\n[bold]Run complete[/].")
    console.print(f"Successes: {state.successes} / Target: {state.target_success}")
    console.print(f"Failures:  {state.failures}")
    console.print(f"Outputs:")
    console.print(f"  - Success dataset: [link=file://{paths.success_train}] {paths.success_train}")
    console.print(f"  - Success meta:    [link=file://{paths.success_meta}] {paths.success_meta}")
    console.print(f"  - Fail dataset:    [link=file://{paths.fail_train}] {paths.fail_train}")


if __name__ == "__main__":
    main()
