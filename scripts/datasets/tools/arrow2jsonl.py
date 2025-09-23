#!/usr/bin/env python3
"""
Convert an Arrow dataset (directory) to JSONL.

Supports:
  - Hugging Face Datasets saved with `save_to_disk` (uses `datasets` if available).
  - Plain folders containing one or more `.arrow` shards (Arrow IPC files).

Default behavior:
  - Input: a directory path that contains the Arrow dataset.
  - Output: JSONL file(s) written to the *same directory* (unless -o/--output is provided).
    * If multiple splits (e.g., train/validation/test) are detected, one JSONL per split.
    * Otherwise a single `dataset.jsonl` is produced.

Examples:
  $ python arrow_to_jsonl.py /path/to/dataset_dir
  $ python arrow_to_jsonl.py /path/to/dataset_dir -o /path/to/output_dir
  $ python arrow_to_jsonl.py /path/to/dataset_dir -o /path/to/output_dir/myfile.jsonl
  $ python arrow_to_jsonl.py /path/to/dataset_dir --batch-size 5000 --force

Requirements:
  - pyarrow (always required)
  - datasets (optional; used automatically if present for HF `save_to_disk` folders)
"""

from __future__ import annotations

import argparse
import base64
import datetime as _dt
import decimal as _dec
import io
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

# Optional imports
try:
    import numpy as _np  # type: ignore
except Exception:  # pragma: no cover
    _np = None

try:
    import datasets as _hfds  # type: ignore
except Exception:  # pragma: no cover
    _hfds = None

import pyarrow as pa
import pyarrow.ipc as pa_ipc


# -----------------------------
# Utilities
# -----------------------------
_SPLIT_ALIASES = {
    "train": "train",
    "test": "test",
    "validation": "validation",
    "valid": "validation",
    "val": "validation",
    "dev": "validation",
}


def _normalize_split(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    name_low = name.lower()
    return _SPLIT_ALIASES.get(name_low, name_low)


def _default_json(o):
    """A robust JSON serializer for Arrow/HF datasets values."""
    # numpy scalars
    if _np is not None:
        if isinstance(o, (_np.integer,)):
            return int(o)
        if isinstance(o, (_np.floating,)):
            # handle NaN/Inf -> JSON doesn't support; convert to None or string
            if _np.isnan(o) or _np.isinf(o):
                return None
            return float(o)
        if isinstance(o, (_np.bool_,)):
            return bool(o)
        if isinstance(o, (_np.ndarray,)):
            return o.tolist()

    # datetime/date/time
    if isinstance(o, (_dt.datetime, _dt.date, _dt.time)):
        # Use ISO 8601; strip tzinfo if present but keep ISO form
        try:
            return o.isoformat()
        except Exception:
            return str(o)

    # decimal -> float (lossy) or str; choose float for convenience
    if isinstance(o, _dec.Decimal):
        try:
            return float(o)
        except Exception:
            return str(o)

    # bytes -> try UTF-8, else base64
    if isinstance(o, (bytes, bytearray, memoryview)):
        try:
            return bytes(o).decode("utf-8")
        except Exception:
            return base64.b64encode(bytes(o)).decode("ascii")

    # sets/tuples
    if isinstance(o, set):
        return list(o)
    if isinstance(o, tuple):
        return list(o)

    # Anything else: fall back to string
    return str(o)


def _jsonl_write_rows(out_fp, rows_iter: Iterable[dict]):
    dumps = json.dumps
    for row in rows_iter:
        out_fp.write(dumps(row, ensure_ascii=False, default=_default_json))
        out_fp.write("\n")


# -----------------------------
# Arrow shard reading (no HF)
# -----------------------------
def _open_arrow_reader(path: Path):
    """
    Try Arrow file reader first; if it fails, try stream reader.
    Returns (reader, mode, closer), where mode is 'file' or 'stream',
    and closer is a context manager-like object to be closed after use.
    """
    # First try file format via memory map (fast, random access).
    try:
        mm = pa.memory_map(str(path), "r")
        reader = pa_ipc.open_file(mm)
        return reader, "file", mm
    except Exception:
        try:
            # Fall back to stream reader via OSFile (sequential).
            osf = pa.OSFile(str(path), "r")
            reader = pa_ipc.open_stream(osf)
            return reader, "stream", osf
        except Exception as e:
            raise RuntimeError(f"Failed to open Arrow reader for {path}: {e}") from e


def _iter_rows_from_record_batch(batch: pa.RecordBatch, max_chunksize: int) -> Iterator[dict]:
    """
    Convert a RecordBatch to rows in manageable chunks to limit memory usage.
    """
    table = pa.Table.from_batches([batch])
    # Re-chunk the table to desired row slices
    total = table.num_rows
    if total == 0:
        return
    cols = table.to_pydict()  # dict[str, List[Any]]
    keys = list(cols.keys())
    # Create iterators that yield rows without materializing all rows again
    # We traverse in blocks to reduce Python overhead on extremely large batches
    for start in range(0, total, max_chunksize):
        end = min(total, start + max_chunksize)
        # Slice columns once
        sliced_cols = {k: v[start:end] for k, v in cols.items()}
        block_len = end - start
        for i in range(block_len):
            yield {k: sliced_cols[k][i] for k in keys}


def _iter_rows_from_arrow_file(shard_path: Path, batch_size: int) -> Iterator[dict]:
    """
    Stream rows from a single .arrow shard path.
    """
    reader, mode, closer = _open_arrow_reader(shard_path)
    try:
        if mode == "file":
            # RecordBatchFileReader
            for i in range(reader.num_record_batches):
                batch = reader.get_batch(i)
                yield from _iter_rows_from_record_batch(batch, batch_size)
        else:
            # RecordBatchStreamReader
            while True:
                try:
                    batch = reader.read_next_batch()
                except StopIteration:
                    break
                yield from _iter_rows_from_record_batch(batch, batch_size)
    finally:
        # Ensure underlying file is closed
        try:
            closer.close()
        except Exception:
            pass


# -----------------------------
# HF Datasets reading (if available)
# -----------------------------
def _hf_is_dataset_dir(path: Path) -> bool:
    if _hfds is None:
        return False
    # Heuristic: dirs created by save_to_disk typically contain these
    markers = {"dataset_info.json", "state.json", "dataset_dict.json"}
    present = any((path / m).exists() for m in markers)
    # A more robust check: try load_from_disk
    if present:
        return True
    # If not obvious, we can conservatively try to load and catch
    try:
        _ = _hfds.load_from_disk(str(path))
        return True
    except Exception:
        return False


def _hf_iter_rows(ds, batch_size: int) -> Iterator[dict]:
    """
    Iterate rows from a Hugging Face Dataset in Python-native form.
    """
    # Ensure Python-native types; this helps JSON serialization
    try:
        ds = ds.with_format("python")
    except Exception:
        try:
            ds.set_format(type="python")
        except Exception:
            pass

    # datasets has batched iteration via .iter or .to_iterable_dataset
    try:
        # .to_iterable_dataset shuffles into streaming form; we keep same order
        for example in ds:
            yield example
    except Exception:
        # Fallback: manual slicing in chunks
        n = len(ds)
        for start in range(0, n, batch_size):
            end = min(n, start + batch_size)
            batch = ds[start:end]
            for ex in batch:
                yield ex


# -----------------------------
# Dataset discovery in a folder of arrow shards
# -----------------------------
def _detect_arrow_groups(root: Path) -> Dict[str, List[Path]]:
    """
    Group .arrow shards by split where possible.

    Heuristics:
      - If shards live under subfolders (e.g., root/train/*.arrow), use that subfolder name as split.
      - Else, try to infer from filename prefix (train*, test*, val*, dev*, validation*).
      - Else, group under 'all'.
    """
    groups: Dict[str, List[Path]] = {}
    all_shards = sorted(root.rglob("*.arrow"))
    if not all_shards:
        return groups

    for p in all_shards:
        # skip index or cache internals if any
        if any(part in {".arrow_cache", "__pycache__"} for part in p.parts):
            continue

        rel = p.relative_to(root)
        split = None

        if len(rel.parts) > 1 and rel.parts[0] not in {"data"}:
            split = _normalize_split(rel.parts[0])
        else:
            # Extract split from filename prefix
            m = re.match(r"^(train|test|validation|valid|val|dev)[\-. _]", p.name, flags=re.I)
            if m:
                split = _normalize_split(m.group(1))

        if not split:
            split = "all"

        groups.setdefault(split, []).append(p)

    return groups


# -----------------------------
# Output path handling
# -----------------------------
def _resolve_output_paths(
    input_dir: Path,
    groups: Dict[str, List[Path]],
    user_output: Optional[Path],
) -> Dict[str, Path]:
    """
    Decide output JSONL path(s) for split groups.
    Rules:
      - If user_output is a directory, write <split>.jsonl (or dataset.jsonl if only 'all').
      - If user_output is a file:
          * If one group -> use exactly that file.
          * If multiple groups -> append _<split> before extension.
      - If no user_output:
          * Write to input_dir with same rules (dataset.jsonl if one group 'all',
            else <split>.jsonl each).
    """
    multiple_groups = len(groups) > 1 or (len(groups) == 1 and next(iter(groups)) != "all")

    if user_output:
        if user_output.exists() and user_output.is_dir():
            out_dir = user_output
            out_dir.mkdir(parents=True, exist_ok=True)
            if not multiple_groups:
                return {"all": out_dir / "dataset.jsonl"}
            else:
                return {split: out_dir / f"{split}.jsonl" for split in groups.keys()}
        else:
            # Treat as file path
            if not multiple_groups:
                # single group -> exact file
                return {next(iter(groups.keys()), "all"): user_output}
            else:
                stem = user_output.stem
                suffix = user_output.suffix or ".jsonl"
                parent = user_output.parent
                parent.mkdir(parents=True, exist_ok=True)
                return {split: parent / f"{stem}_{split}{suffix}" for split in groups.keys()}
    else:
        out_dir = input_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        if not multiple_groups:
            return {"all": out_dir / "dataset.jsonl"}
        else:
            return {split: out_dir / f"{split}.jsonl" for split in groups.keys()}


# -----------------------------
# Main conversion flows
# -----------------------------
def convert_hf_dataset_dir(
    input_dir: Path,
    output: Optional[Path],
    batch_size: int,
    force: bool,
) -> None:
    obj = _hfds.load_from_disk(str(input_dir))
    # Either a Dataset or a DatasetDict
    if hasattr(obj, "keys") and hasattr(obj, "__getitem__"):
        # DatasetDict-like
        groups = {k: [k] for k in obj.keys()}  # map split->placeholder
        out_map = _resolve_output_paths(input_dir, groups, output)
        for split, _ in groups.items():
            out_path = out_map[split]
            if out_path.exists() and not force:
                raise FileExistsError(f"{out_path} exists; use --force to overwrite.")
            ds_split = obj[split]
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", encoding="utf-8") as fp:
                _jsonl_write_rows(fp, _hf_iter_rows(ds_split, batch_size))
            print(f"Wrote {split} -> {out_path}")
    else:
        # Single Dataset
        groups = {"all": ["all"]}
        out_map = _resolve_output_paths(input_dir, groups, output)
        out_path = out_map["all"]
        if out_path.exists() and not force:
            raise FileExistsError(f"{out_path} exists; use --force to overwrite.")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fp:
            _jsonl_write_rows(fp, _hf_iter_rows(obj, batch_size))
        print(f"Wrote {out_path}")


def convert_arrow_shards_dir(
    input_dir: Path,
    output: Optional[Path],
    batch_size: int,
    force: bool,
) -> None:
    groups = _detect_arrow_groups(input_dir)
    if not groups:
        raise FileNotFoundError(f"No .arrow shards found under: {input_dir}")
    out_map = _resolve_output_paths(input_dir, groups, output)

    for split, files in groups.items():
        files = sorted(files)
        out_path = out_map[split if split in out_map else "all"]
        if out_path.exists() and not force:
            raise FileExistsError(f"{out_path} exists; use --force to overwrite.")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        with out_path.open("w", encoding="utf-8") as fp:
            for shard in files:
                for row in _iter_rows_from_arrow_file(shard, batch_size):
                    _jsonl_write_rows(fp, (row for row in [row]))  # write row-by-row
                    written += 1
        print(f"Wrote {written} rows from {len(files)} shard(s) -> {out_path}")


# -----------------------------
# CLI
# -----------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert an Arrow dataset directory to JSONL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "input_dir",
        type=str,
        help="Directory containing the Arrow dataset (HF save_to_disk or *.arrow shards).",
    )
    p.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output directory or file path for the JSONL result(s). "
             "If directory: creates <split>.jsonl (or dataset.jsonl). "
             "If file path: uses exactly that file when single split; appends _<split> otherwise.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=10000,
        help="Max rows processed per internal batch to limit memory usage.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output file(s).",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"ERROR: input path must be an existing directory: {input_dir}", file=sys.stderr)
        return 2

    out_path = Path(args.output).expanduser().resolve() if args.output else None

    # If HF datasets is available and this looks like a HF dataset folder, use it.
    if _hf_is_dataset_dir(input_dir):
        try:
            convert_hf_dataset_dir(input_dir, out_path, args.batch_size, args.force)
            return 0
        except Exception as e:
            print(f"WARNING: Hugging Face path detected but failed to convert via datasets: {e}", file=sys.stderr)
            print("Falling back to raw Arrow shard conversion...", file=sys.stderr)

    # Fallback: treat as a folder of Arrow shards.
    try:
        convert_arrow_shards_dir(input_dir, out_path, args.batch_size, args.force)
        return 0
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
