#!/usr/bin/env python3
import argparse
import json
import os
import random
from pathlib import Path

from datasets import load_from_disk, load_dataset, DatasetDict, Dataset
from rich import box
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

def parse_args():
    parser = argparse.ArgumentParser(description="Inspect a HuggingFace dataset from a folder")
    parser.add_argument(
        "--folder", type=str, required=True, help="Folder containing the dataset."
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=2,
        help="How many random samples to print from each split (default: 2).",
    )
    parser.add_argument(
        "--count_chars",
        action="store_true",
        help="If set, count characters in each column (default: False).",
    )
    return parser.parse_args()

def load_dataset_from_folder(folder: Path):
    """
    Load a dataset from folder. If folder contains a saved HF dataset (dataset_info.json and state.json
    or dataset_dict.json), use load_from_disk. Otherwise, check if folder contains parquet, jsonl or txt files
    and use load_dataset with appropriate arguments.
    """
    folder = folder.resolve()
    info = {}
    info_files = {
        "dataset_info": folder / "dataset_info.json",
        "state": folder / "state.json",
        "dataset_dict": folder / "dataset_dict.json",
    }
    # Check for saved dataset metadata
    if info_files["dataset_info"].exists() and info_files["state"].exists():
        try:
            with info_files["dataset_info"].open("r", encoding="utf-8") as f:
                info["dataset_info"] = json.load(f)
            with info_files["state"].open("r", encoding="utf-8") as f:
                info["state"] = json.load(f)
        except Exception as e:
            console.print(f"[red]Error reading dataset_info/state files:[/red] {e}")
    elif info_files["dataset_dict"].exists():
        try:
            with info_files["dataset_dict"].open("r", encoding="utf-8") as f:
                info["dataset_dict"] = json.load(f)
        except Exception as e:
            console.print(f"[red]Error reading dataset_dict.json:[/red] {e}")

    # If we found one of these metadata files, assume it's a saved HF dataset.
    if info:
        console.print("[green]Found dataset metadata files. Using load_from_disk.[/green]")
        ds = load_from_disk(str(folder))
        return ds, info
    else:
        # Otherwise check for files with known extensions.
        for ext, hf_builder in (("parquet", "parquet"), ("jsonl", "json"), ("txt", "text")):
            files = list(folder.glob(f"*.{ext}"))
            if files:
                data_files = [str(f) for f in files]
                console.print(
                    f"[green]Found {len(data_files)} .{ext} file(s). Loading with load_dataset('{hf_builder}', data_files=...)[/green]"
                )
                ds = load_dataset(hf_builder, data_files=data_files)
                return ds, {}
        # Fallback: try load_from_disk anyway.
        try:
            console.print("[yellow]No metadata or recognized files found; trying load_from_disk.[/yellow]")
            ds = load_from_disk(str(folder))
            return ds, {}
        except Exception as e:
            console.print(f"[red]Failed to load dataset from folder: {e}[/red]")
            exit(1)

def print_dataset_info(info: dict):
    """Print any useful metadata from dataset_info.json/state.json or dataset_dict.json."""
    if not info:
        console.print(Panel("[italic](No dataset metadata available)[/italic]", title="Dataset Info"))
        return

    table = Table(title="Dataset Metadata", box=box.SIMPLE_HEAVY)
    table.add_column("Key", style="bold cyan")
    table.add_column("Value", style="magenta")
    for key, content in info.items():
        # Dump content as pretty JSON (or part of it) for display.
        content_str = json.dumps(content, indent=2)
        table.add_row(key, content_str)
    console.print(table)

def sample_split(ds_split, split_name: str, num_samples: int):
    """Select num_samples random rows from ds_split."""
    total = len(ds_split)
    if total == 0:
        return []
    indices = random.sample(range(total), min(num_samples, total))
    samples = ds_split.select(indices)
    return samples

def print_samples_table(samples: Dataset, split_name: str):
    """Print sample rows from a split in a table."""
    if len(samples) == 0:
        console.print(f"[red]No samples to show for split {split_name}.[/red]")
        return

    table = Table(title=f"Random Samples from split [bold]{split_name}[/bold]", box=box.MINIMAL_DOUBLE_HEAD)
    # Add a column for the row index
    table.add_column("Row #", style="yellow", justify="right")
    # Use the keys from the first sample as columns
    keys = list(samples[0].keys())
    for key in keys:
        table.add_column(key, style="cyan")
    for idx, row in enumerate(samples):
        # For each row, display row index and the value for each key.
        row_values = [str(row.get(k, "")) for k in keys]
        table.add_row(str(idx), *row_values)
    console.print(table)

def count_chars_in_split(ds_split):
    """
    For each column in ds_split, count total characters.
    Returns a dict mapping column name to total char count.
    """
    totals = {}
    keys = ds_split.column_names
    # Initialize totals to zero
    for k in keys:
        totals[k] = 0
    # We iterate in batches for some efficiency (this may still be expensive for huge datasets).
    for row in ds_split:
        for k in keys:
            value = row.get(k, "")
            totals[k] += len(str(value))
    return totals

def print_split_statistics(splits: dict, count_chars: bool, samples: int):
    """
    For each split (a dict of {split_name: dataset}):
      1. Print number of rows and columns.
      2. Print a table of random samples.
      3. If count_chars is True, count characters per column.
    Also accumulate grand totals across splits.
    """
    grand_rows = 0
    grand_char_totals = {}  # key: column name, value: total char count
    for split_name, ds_split in splits.items():
        num_rows = len(ds_split)
        grand_rows += num_rows
        cols = ds_split.column_names

        console.rule(f"[bold blue]Split: {split_name}[/bold blue]")

        # Print basic info
        info_table = Table(box=box.MINIMAL)
        info_table.add_column("Property", style="bold")
        info_table.add_column("Value", style="magenta")
        info_table.add_row("Rows", f"{num_rows:,}")
        info_table.add_row("Columns", ", ".join(cols))
        console.print(info_table)

        # Print random samples
        samples_ds = sample_split(ds_split, split_name, samples)
        print_samples_table(samples_ds, split_name)

        # Count characters if requested
        if count_chars:
            console.print("[bold green]Counting characters per column...[/bold green]")
            char_totals = count_chars_in_split(ds_split)
            # Accumulate to grand totals. We use union of columns.
            for k, v in char_totals.items():
                grand_char_totals[k] = grand_char_totals.get(k, 0) + v

            char_table = Table(title=f"Character Count for Split: {split_name}", box=box.SIMPLE)
            char_table.add_column("Column", style="cyan", justify="left")
            char_table.add_column("Char Count", style="magenta", justify="right")
            split_total = 0
            for k in cols:
                cnt = char_totals.get(k, 0)
                split_total += cnt
                char_table.add_row(k, f"{cnt:,}")
            char_table.add_row("[bold]Grand Total[/bold]", f"[bold]{split_total:,}[/bold]")
            console.print(char_table)
        else:
            console.print("[italic yellow]Character counting skipped.[/italic yellow]")

    # After all splits, print grand totals
    console.rule("[bold red]Grand Totals Across All Splits[/bold red]")
    grand_table = Table(box=box.HEAVY_EDGE)
    grand_table.add_column("Metric", style="bold cyan")
    grand_table.add_column("Value", style="bold magenta", justify="right")
    grand_table.add_row("Total Rows", f"{grand_rows:,}")
    if count_chars:
        # Sum all char totals across columns
        overall_chars = sum(grand_char_totals.values())
        details = ", ".join(f"{k}: {v:,}" for k, v in grand_char_totals.items())
        grand_table.add_row("Total Characters (per col)", details)
        grand_table.add_row("Grand Total Characters", f"{overall_chars:,}")
    else:
        grand_table.add_row("Character Counting", "(skipped)")
    console.print(grand_table)

def print_file_listing(folder: Path):
    """
    List files in the folder (recursively) that are part of the dataset, and print their sizes.
    """
    folder = folder.resolve()
    file_table = Table(title="Dataset Files", box=box.MINIMAL_DOUBLE_HEAD)
    file_table.add_column("File", style="cyan")
    file_table.add_column("Size (bytes)", style="magenta", justify="right")
    total_size = 0
    # Search recursively for files (you can adjust the pattern if needed)
    for file_path in folder.rglob("*"):
        if file_path.is_file():
            try:
                size = file_path.stat().st_size
            except Exception:
                size = 0
            total_size += size
            # Show relative path from folder
            rel_path = str(file_path.relative_to(folder))
            file_table.add_row(rel_path, f"{size:,}")
    file_table.add_row("[bold]Grand Total[/bold]", f"[bold]{total_size:,}[/bold]")
    console.print(file_table)

def main():
    args = parse_args()
    folder = Path(args.folder)
    if not folder.exists():
        console.print(f"[red]Folder not found: {folder}[/red]")
        exit(1)

    # Load dataset (and possibly metadata)
    ds_loaded, metadata = load_dataset_from_folder(folder)

    # HuggingFace APIs may return either a DatasetDict (multiple splits) or a single Dataset.
    if isinstance(ds_loaded, DatasetDict):
        splits = ds_loaded
    elif isinstance(ds_loaded, Dataset):
        splits = {"default": ds_loaded}
    else:
        console.print("[red]Unknown dataset type returned.[/red]")
        exit(1)

    # Print any metadata available from the saved info files.
    print_dataset_info(metadata)

    # For each split, print info, samples, char counts.
    print_split_statistics(splits, args.count_chars, args.samples)

    # Print list of files in the dataset folder along with file sizes.
    print_file_listing(folder)

if __name__ == "__main__":
    main()
