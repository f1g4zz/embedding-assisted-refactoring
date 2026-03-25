import pandas as pd
import os
import argparse

def enrich_project(smells_path, joern_map_path, embeddings_path, output_path, line_tolerance=10):
    print(f"[*] Loading: {os.path.basename(smells_path)}")
    
    df_smells = pd.read_csv(smells_path)
    df_map = pd.read_csv(joern_map_path)
    df_emb = pd.read_csv(embeddings_path)
    
    df_joern = pd.merge(df_map, df_emb, on="method_id")

    df_smells['file_key'] = df_smells['File'].str.lower().str.replace('\\', '/')
    df_joern['file_key'] = df_joern['filename'].str.lower().str.replace('\\', '/')
    
    results = []
    matches = 0
    misses = 0

    for _, row in df_smells.iterrows():
        candidates = df_joern[
            (df_joern['name'] == row['Method']) & 
            (df_joern['className'] == row['Class']) &
            (df_joern['file_key'].str.contains(row['file_key'], na=False))
        ]
        
        best_match = None
        
        if len(candidates) == 1:
            best_match = candidates.iloc[0]
        elif len(candidates) > 1:

            candidates['line_diff'] = (candidates['lineNumber'] - row['Line no']).abs()
            potential = candidates[candidates['line_diff'] <= line_tolerance]
            
            if not potential.empty:
                best_match = potential.sort_values('line_diff').iloc[0]
        
        if best_match is not None:
            emb_cols = [c for c in df_joern.columns if c.startswith('f_')]
            combined_row = {**row.to_dict(), **best_match[emb_cols].to_dict()}
            combined_row['joern_line'] = best_match['lineNumber'] # Per debug
            results.append(combined_row)
            matches += 1
        else:
            misses += 1

    df_final = pd.DataFrame(results)
    df_final.to_csv(output_path, index=False)
    print(f"[SUCCESS] Match completati: {matches} | Falliti: {misses}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adds LINE's embeddings")
    parser.add_argument("--smells", required=True, help="Path to main CSV")
    parser.add_argument("--map", required=True, help="Path to mapping from Joern")
    parser.add_argument("--emb", required=True, help="Path to LINE's embeddings")
    parser.add_argument("--out", required=True, help="output")
    parser.add_argument("--tol", type=int, default=15, help="Tollerance  (line offset, default 15)")
    
    args = parser.parse_args()
    enrich_project(args.smells, args.map, args.emb, args.out, args.tol)