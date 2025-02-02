import os
import argparse
import json
import pandas as pd
import pyarrow.feather as feather
from datasets import load_dataset
from rich.console import Console
from rich.table import Table

def detect_huggingface_dataset(folder_path):
    """Check if the folder contains Hugging Face dataset files."""
    return os.path.exists(os.path.join(folder_path, "dataset_info.json")) and \
           os.path.exists(os.path.join(folder_path, "state.json"))

def load_file(filepath):
    try:
        if filepath.endswith(".parquet"):
            df = pd.read_parquet(filepath)
        elif filepath.endswith(".arrow"):
            df = feather.read_feather(filepath)
        elif filepath.endswith(".jsonl"):
            df = pd.read_json(filepath, lines=True)
        elif filepath.endswith(".txt"):
            df = pd.read_csv(filepath, sep=None, engine='python')
        else:
            console.print(f"[red]Unsupported file format: {filepath}[/red]")
            return None
        return df
    except Exception as e:
        console.print(f"[red]Failed to read {filepath}: {e}[/red]")
        return None

def get_file_stats(filepath):
    df = load_file(filepath)
    if df is None:
        return None
    num_rows = df.shape[0]
    total_chars = int(df.astype(str).apply(lambda col: col.str.len()).sum().sum())
    columns = list(df.columns)
    file_size = os.path.getsize(filepath)
    return df, num_rows, total_chars, columns, file_size

def print_summary(file_summaries):
    summary_table = Table(title="Dataset Files Summary", show_lines=True)
    summary_table.add_column("File", style="cyan", no_wrap=True)
    summary_table.add_column("Rows", justify="right")
    summary_table.add_column("Total Characters", justify="right")
    summary_table.add_column("File Size (bytes)", justify="right")
    summary_table.add_column("Columns", style="magenta")
    for summary in file_summaries:
        filename, num_rows, total_chars, columns, file_size = summary
        col_str = ", ".join(columns)
        summary_table.add_row(filename, str(num_rows), str(total_chars), str(file_size), col_str)
    console.print(summary_table)

def print_sample(df, filename, sample_size):
    sample_size = min(sample_size, len(df))
    sample_df = df.sample(n=sample_size)
    sample_table = Table(title=f"Random Sample from {filename}", show_lines=True)
    for col in sample_df.columns:
        sample_table.add_column(str(col), style="green", overflow="fold")
    for _, row in sample_df.iterrows():
        sample_table.add_row(*[str(val) for val in row])
    console.print(sample_table)

def process_huggingface_dataset(folder_path, sample_size):
    console.print(f"[yellow]Detected Hugging Face dataset in {folder_path}[/yellow]")
    dataset = load_dataset(folder_path)
    df = pd.DataFrame(dataset["train"][0:1000])
    num_rows = len(df)
    total_chars = int(df.astype(str).apply(lambda col: col.str.len()).sum().sum())
    columns = list(df.columns)

    # Load dataset_info.json and state.json
    with open(os.path.join(folder_path, "dataset_info.json"), "r") as f:
        dataset_info = json.load(f)
    with open(os.path.join(folder_path, "state.json"), "r") as f:
        state_info = json.load(f)

    console.print(f"[blue]Dataset Info:[/blue] {json.dumps(dataset_info, indent=2)}")
    console.print(f"[blue]State Info:[/blue] {json.dumps(state_info, indent=2)}")

    print_sample(df, folder_path, sample_size)
    return (folder_path, num_rows, total_chars, columns, "N/A")

def main():
    parser = argparse.ArgumentParser(description="Summarize dataset files and Hugging Face datasets.")
    parser.add_argument("--folder", required=True, help="Path to dataset folder")
    parser.add_argument("--sample_size", type=int, default=2, help="Number of sample rows to display")
    parser.add_argument("--max_sample_files", type=int, default=1, help="Maximum number of files to display samples from")
    args = parser.parse_args()
    folder_path = args.folder
    file_summaries = []

    if detect_huggingface_dataset(folder_path):
        hf_summary = process_huggingface_dataset(folder_path, args.sample_size)
        file_summaries.append(hf_summary)
    else:
        file_types = [".parquet", ".jsonl", ".txt"]
        selected_files = []
        for file_type in file_types:
            files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(file_type)]
            if files:
                selected_files = files
                break  # Process only the first detected file type

        if selected_files:
            for idx, filepath in enumerate(selected_files):
                result = get_file_stats(filepath)
                if result:
                    df, num_rows, total_chars, columns, file_size = result
                    file_summaries.append((os.path.basename(filepath), num_rows, total_chars, columns, file_size))
                    if idx < args.max_sample_files:
                        print_sample(df, os.path.basename(filepath), args.sample_size)
        else:
            console.print(f"[red]No supported dataset files found in {folder_path}[/red]")

    if file_summaries:
        print_summary(file_summaries)

if __name__ == "__main__":
    console = Console()
    main()
