#!/usr/bin/env python3
import os
import argparse
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

def get_file_stats(filepath):
    """
    Read the parquet file and compute:
      - number of rows,
      - total number of characters across all columns,
      - list of columns.
    """
    try:
        df = pd.read_parquet(filepath)
    except Exception as e:
        console.print(f"[red]Failed to read {filepath}: {e}[/red]")
        return None

    # Number of rows in the DataFrame
    num_rows = df.shape[0]

    # Total number of characters:
    # Convert each value to string, compute its length and then sum over the entire DataFrame.
    # Note: This may be expensive if the DataFrame is very large.
    total_chars = int(df.astype(str).apply(lambda col: col.str.len()).sum().sum())

    # List of columns
    columns = list(df.columns)

    # File size in bytes using the OS file size (the parquet file size on disk)
    file_size = os.path.getsize(filepath)

    return df, num_rows, total_chars, columns, file_size

def print_summary(file_summaries):
    """
    Print a summary table for each file with stats:
      - File Name
      - Number of Rows
      - Total Characters in All Columns
      - File Size (bytes)
      - Columns Available (as comma-separated list)
    """
    summary_table = Table(title="Parquet Files Summary", show_lines=True)
    summary_table.add_column("File", style="cyan", no_wrap=True)
    summary_table.add_column("Rows", justify="right")
    summary_table.add_column("Total Characters", justify="right")
    summary_table.add_column("File Size (bytes)", justify="right")
    summary_table.add_column("Columns", style="magenta")

    for summary in file_summaries:
        filepath, num_rows, total_chars, columns, file_size = summary
        # Get the base name of the file
        filename = os.path.basename(filepath)
        # Format columns as a comma separated string
        col_str = ", ".join(columns)
        summary_table.add_row(filename, str(num_rows), str(total_chars), str(file_size), col_str)

    console.print(summary_table)

def print_sample(df, filename, sample_size):
    """
    Print a random sample of n rows from the DataFrame using Rich table.
    """
    # If the DataFrame has fewer rows than sample_size, sample all rows.
    if sample_size > len(df):
        sample_size = len(df)
    sample_df = df.sample(n=sample_size)

    # Create a table with columns from the DataFrame
    sample_table = Table(title=f"Random Sample from {filename}", show_lines=True)
    for col in sample_df.columns:
        sample_table.add_column(str(col), style="green", overflow="fold")

    # Add each row to the table, converting values to string
    for _, row in sample_df.iterrows():
        sample_table.add_row(*[str(val) for val in row])
    console.print(sample_table)

def main():
    parser = argparse.ArgumentParser(description="Process and summarize Parquet files.")
    parser.add_argument("--folder", required=True, help="Path to the folder containing .parquet files")
    parser.add_argument("--sample_size", type=int, nargs="?", default=2, help="Number of random sample rows to display from each file (default: 5)")
    args = parser.parse_args()

    folder_path = args.folder
    sample_size = args.sample_size

    # Gather all .parquet files in the provided folder
    parquet_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.parquet')]
    if not parquet_files:
        console.print(f"[red]No .parquet files found in {folder_path}[/red]")
        return

    file_summaries = []
    for filepath in parquet_files:
        result = get_file_stats(filepath)
        if result is None:
            continue
        df, num_rows, total_chars, columns, file_size = result
        file_summaries.append((filepath, num_rows, total_chars, columns, file_size))
        # Print random sample for the file
        print_sample(df, os.path.basename(filepath), sample_size)

    # Print summary table at the end
    print_summary(file_summaries)

if __name__ == "__main__":
    # Initialize the rich console
    console = Console()
    main()
