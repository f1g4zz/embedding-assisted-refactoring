import pandas as pd
import os
import glob
import re
import argparse
import sys
import csv

def merge_chunks(directory_path, project_name):
    csv.field_size_limit(2147483647)
    search_pattern = os.path.join(directory_path, f"{project_name}_chunk*")
    files = glob.glob(search_pattern)
    
    print(f"DEBUG: Cerco in: {directory_path}")
    print(f"DEBUG: Pattern usato: {project_name}_chunk*")

    if not files:
        print(f"\nERRORE: Nessun file trovato!")
        print(f"Controlla che i file inizino esattamente con: {project_name}_chunk")
        return

    files.sort(key=lambda x: [int(c) if c.isdigit() else c for c in re.split('([0-9]+)', x)])

    try:
        with open(files[0], 'r') as f:
            first_line = f.readline()
            separator = '|' if '|' in first_line else ' '
    except Exception as e:
        print(f"Errore lettura file: {e}")
        return
    
    print(f"\n--- Info ---")
    print(f"File trovati: {len(files)}")
    print(f"Separatore: '{'PIPE' if separator == '|' else 'SPAZIO'}'")
    
    all_dfs = []
    for file in files:
        df = pd.read_csv(file, sep=separator, header=None, engine='python', dtype=str)
        all_dfs.append(df)

    combined_df = pd.concat(all_dfs, ignore_index=True)

    combined_df[0] = range(1, len(combined_df) + 1)

    output_filename = os.path.join(directory_path, f"{project_name}.csv")
    combined_df.to_csv(output_filename, sep=separator, index=False, header=False, quoting=3, escapechar=" ")
    
    print(f"--- Risultato ---")
    print(f"Creato: {output_filename}")
    print(f"Righe totali: {len(combined_df)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", required=True)
    parser.add_argument("-p", "--project", required=True)
    args = parser.parse_args()

    if not os.path.isdir(args.dir):
        print(f"Errore: La cartella '{args.dir}' non esiste.")
        sys.exit(1)

    merge_chunks(args.dir, args.project)