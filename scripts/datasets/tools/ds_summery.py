#!/usr/bin/env python
import argparse
import os
import json
import random
import pandas as pd
from rich.console import Console
from rich.table import Table

def process_hf_dataset(folder, samples):
    console = Console()
    console.print(f"\n[bold green]HuggingFace Dataset detected in folder:[/bold green] {folder}")
    hf_dict_path = os.path.join(folder, "dataset_dict.json")
    if os.path.exists(hf_dict_path):
        # Process dataset with splits
        with open(hf_dict_path, "r", encoding="utf-8") as f:
            dataset_dict = json.load(f)
        table = Table(title="HuggingFace Dataset Splits")
        table.add_column("Split", style="cyan")
        table.add_column("Number of Rows", style="magenta", justify="right")
        for split, data in dataset_dict.items():
            nrows = len(data) if isinstance(data, list) else "N/A"
            table.add_row(split, f"{nrows:,}" if isinstance(nrows, int) else str(nrows))
        console.print(table)

        # Sample from the first valid split (list of dicts)
        first_split = next((k for k, v in dataset_dict.items() if isinstance(v, list) and len(v) > 0), None)
        if first_split is not None:
            data = dataset_dict[first_split]
            sample_n = min(samples, len(data))
            sample_rows = random.sample(data, sample_n)
            sample_table = Table(title=f"Random Samples from split '{first_split}'")
            keys = list(sample_rows[0].keys())
            for key in keys:
                sample_table.add_column(key, style="white")
            for row in sample_rows:
                sample_table.add_row(*[str(row.get(k, "")) for k in keys])
            console.print(sample_table)
        else:
            console.print("[red]No sample data available in any split.[/red]")
    else:
        # Otherwise load dataset_info.json and state.json if available.
        info_path = os.path.join(folder, "dataset_info.json")
        state_path = os.path.join(folder, "state.json")
        if os.path.exists(info_path):
            with open(info_path, "r", encoding="utf-8") as f:
                info = json.load(f)
            console.print("[bold blue]dataset_info.json[/bold blue]:")
            console.print_json(json.dumps(info, indent=2))
        if os.path.exists(state_path):
            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            console.print("[bold blue]state.json[/bold blue]:")
            console.print_json(json.dumps(state, indent=2))

def process_single_folder(folder, samples, count_chars):
    """
    Process a single folder (non-recursively) that is assumed to be a non-HuggingFace dataset.
    It detects the file type (priority: .parquet > .jsonl > .txt) from immediate files,
    prints file statistics and random samples in colored tables,
    and returns a summary dictionary.
    """
    console = Console()
    console.print(f"\n[bold underline]Processing folder:[/bold underline] {folder}")

    # Get immediate files in the folder
    try:
        files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    except Exception as e:
        console.print(f"[red]Error listing files in {folder}: {e}[/red]")
        return None

    file_ext = None
    for ext in [".parquet", ".jsonl", ".txt"]:
        if any(f.endswith(ext) for f in files):
            file_ext = ext
            break

    if file_ext is None:
        console.print(f"[red]No dataset files (.parquet, .jsonl, .txt) found in folder:[/red] {folder}")
        return {
            "folder": folder,
            "rows": 0,
            "chars": 0 if count_chars else "(skipped)",
            "bytes": 0
        }

    # Gather file paths (only immediate files)
    file_paths = [os.path.join(folder, f) for f in files if f.endswith(file_ext)]
    all_samples = []
    stats = []
    total_rows = 0
    total_chars = 0
    total_bytes = 0

    for file in file_paths:
        try:
            file_size = os.path.getsize(file)
        except Exception as e:
            console.print(f"[red]Error getting size for {file}: {e}[/red]")
            continue
        total_bytes += file_size

        try:
            if file_ext == ".parquet":
                df = pd.read_parquet(file)
            elif file_ext == ".jsonl":
                df = pd.read_json(file, lines=True)
            elif file_ext == ".txt":
                with open(file, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                df = pd.DataFrame({"text": [line.strip() for line in lines]})
            else:
                continue
        except Exception as e:
            console.print(f"[red]Error reading {file}: {e}[/red]")
            continue

        nrows = len(df)
        total_rows += nrows

        if count_chars:
            try:
                file_chars = df.astype(str).applymap(len).sum().sum()
            except Exception:
                file_chars = 0
            total_chars += file_chars
        else:
            file_chars = "(skipped)"

        columns_list = list(df.columns)
        stats.append({
            "File": file,
            "Rows": nrows,
            "Chars": file_chars,
            "Byte Size": file_size,
            "Columns": ", ".join(columns_list)
        })

        # Sample rows from this file if available
        if nrows > 0:
            sample_n = min(samples, nrows)
            sample_df = df.sample(n=sample_n)
            sample_df["Source File"] = os.path.basename(file)
            all_samples.append(sample_df)

    # Print file statistics table for this folder
    stats_table = Table(title=f"File Statistics: {folder}", show_footer=True)
    stats_table.add_column("File", style="cyan")
    stats_table.add_column("Rows", style="magenta", justify="right", footer=f"{total_rows:,}")
    stats_table.add_column("Chars", style="green", justify="right", footer=f"{total_chars:,}" if count_chars else "(skipped)")
    stats_table.add_column("Byte Size", style="yellow", justify="right", footer=f"{total_bytes:,}")
    stats_table.add_column("Columns", style="blue")
    for stat in stats:
        rows_str = f"{stat['Rows']:,}"
        chars_str = f"{stat['Chars']:,}" if (isinstance(stat["Chars"], int)) else stat["Chars"]
        bytes_str = f"{stat['Byte Size']:,}"
        stats_table.add_row(stat["File"], rows_str, chars_str, bytes_str, stat["Columns"])
    console.print(stats_table)

    # Print sample rows table if available
    if all_samples:
        samples_table = Table(title=f"Random Samples: {folder}")
        combined_samples = pd.concat(all_samples, ignore_index=True)
        if len(combined_samples) > samples:
            combined_samples = combined_samples.sample(n=samples)
        for col in combined_samples.columns:
            samples_table.add_column(col, style="white")
        for _, row in combined_samples.iterrows():
            samples_table.add_row(*[str(row[col]) for col in combined_samples.columns])
        console.print(samples_table)
    else:
        console.print("[red]No sample rows to display.[/red]")

    return {
        "folder": folder,
        "rows": total_rows,
        "chars": total_chars if count_chars else None,
        "bytes": total_bytes
    }

def process_recursive_non_hf(root_folder, samples, count_chars):
    console = Console()
    summary_list = []
    # Walk through all directories starting at root_folder.
    # Each directory is processed individually (only immediate files).
    for current_folder, dirs, files in os.walk(root_folder):
        # Skip folders that appear to be HuggingFace datasets.
        if (os.path.exists(os.path.join(current_folder, "dataset_info.json")) and
            os.path.exists(os.path.join(current_folder, "state.json"))) or \
           os.path.exists(os.path.join(current_folder, "dataset_dict.json")):
            console.print(f"\n[bold yellow]Skipping HuggingFace dataset folder:[/bold yellow] {current_folder}")
            continue
        summary = process_single_folder(current_folder, samples, count_chars)
        if summary:
            summary_list.append(summary)
    # After processing all folders, build a summary table.
    if summary_list:
        grand_total_rows = sum(s["rows"] for s in summary_list)
        grand_total_bytes = sum(s["bytes"] for s in summary_list)
        if count_chars:
            grand_total_chars = sum(s["chars"] for s in summary_list if isinstance(s["chars"], int))
        else:
            grand_total_chars = "(skipped)"
        sum_table = Table(title="Summary of Folder Totals", show_footer=True)
        sum_table.add_column("Folder", style="cyan")
        sum_table.add_column("Total Rows", style="magenta", justify="right")
        sum_table.add_column("Total Chars", style="green", justify="right")
        sum_table.add_column("Total Bytes", style="yellow", justify="right")
        for s in summary_list:
            rows_str = f"{s['rows']:,}"
            chars_str = f"{s['chars']:,}" if (count_chars and isinstance(s["chars"], int)) else (s["chars"] if not count_chars else "0")
            bytes_str = f"{s['bytes']:,}"
            sum_table.add_row(s["folder"], rows_str, chars_str, bytes_str)
        # Grand-grand total row as footer
        sum_table.columns[1].footer = f"{grand_total_rows:,}"
        sum_table.columns[2].footer = f"{grand_total_chars:,}" if count_chars and isinstance(grand_total_chars, int) else grand_total_chars
        sum_table.columns[3].footer = f"{grand_total_bytes:,}"
        console.print(sum_table)

def main():
    parser = argparse.ArgumentParser(
        description="Process a dataset folder (HuggingFace dataset or set of parquet/jsonl/txt files) and display info."
    )
    parser.add_argument("--folder", type=str, required=True, help="Folder to look into")
    parser.add_argument("--samples", type=int, default=2, help="How many random samples to print from the dataset")
    parser.add_argument("--count_chars", action="store_true", help="Count characters in dataset")
    parser.add_argument("--recursive", action="store_true", help="Recursively process child folders (if non-HuggingFace)")
    args = parser.parse_args()

    folder = args.folder
    # Check if the root folder is a HuggingFace dataset.
    is_hf = ((os.path.exists(os.path.join(folder, "dataset_info.json")) and
              os.path.exists(os.path.join(folder, "state.json"))) or
             os.path.exists(os.path.join(folder, "dataset_dict.json")))
    if is_hf:
        process_hf_dataset(folder, args.samples)
    else:
        if args.recursive:
            process_recursive_non_hf(folder, args.samples, args.count_chars)
        else:
            process_single_folder(folder, args.samples, args.count_chars)

if __name__ == "__main__":
    main()
