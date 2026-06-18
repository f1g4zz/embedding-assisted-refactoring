import pandas as pd
import os
import glob
import re
import argparse
import sys
import csv

def merge_chunks(directory_path, project_name):
    """
    Merges chunk files generated for a project into a single CSV.
    """
    csv.field_size_limit(2147483647)
    search_pattern = os.path.join(directory_path, f"{project_name}_chunk*")
    files = glob.glob(search_pattern)
    
    print(f"DEBUG: Searching in: {directory_path}")
    print(f"DEBUG: Pattern used: {project_name}_chunk*")

    if not files:
        print(f"\nERROR: No files found!")
        print(f"Ensure that files start exactly with: {project_name}_chunk")
        return

    # Sort files naturally by chunk index
    files.sort(key=lambda x: [int(c) if c.isdigit() else c for c in re.split('([0-9]+)', x)])

    try:
        with open(files[0], 'r') as f:
            first_line = f.readline()
            separator = '|' if '|' in first_line else ' '
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    print(f"\n--- Info ---")
    print(f"Files found: {len(files)}")
    print(f"Separator: '{'PIPE' if separator == '|' else 'SPACE'}'")
    
    all_dfs = []
    for file in files:
        df = pd.read_csv(file, sep=separator, header=None, engine='python', dtype=str)
        all_dfs.append(df)

    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Re-index the first column to match the combined row indices
    combined_df[0] = range(1, len(combined_df) + 1)

    output_filename = os.path.join(directory_path, f"{project_name}.csv")
    combined_df.to_csv(output_filename, sep=separator, index=False, header=False, quoting=3, escapechar=" ")
    
    print(f"--- Result ---")
    print(f"Created: {output_filename}")
    print(f"Total rows: {len(combined_df)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", required=True)
    parser.add_argument("-p", "--project", required=True)
    args = parser.parse_args()

    if not os.path.isdir(args.dir):
        print(f"Error: The folder '{args.dir}' does not exist.")
        sys.exit(1)

    merge_chunks(args.dir, args.project)
