import pandas as pd
import numpy as np
import argparse
import os
import sys
from pathlib import Path

def normalize_path(path):
    if pd.isna(path): return ""
    return str(path).replace('\\', '/').lower().strip()

def apply_prefixes(df, columns_to_prefix, prefix):
    rename_dict = {}
    for col in df.columns:
        match = next((c for c in columns_to_prefix if c.lower() == col.lower()), None)
        if match:
            rename_dict[col] = "{}{}".format(prefix, col)
    return df.rename(columns=rename_dict)

def enrich_dataset(main_path, methods_path, classes_path, out_matches, out_track):
    print("Process started for: " + os.path.basename(main_path))
    sys.stdout.flush()
    
    df_main = pd.read_csv(main_path)
    df_methods = pd.read_csv(methods_path)
    df_classes = pd.read_csv(classes_path)

    df_main.columns = [c.strip() for c in df_main.columns]
    
    emb_cols = [c for c in df_main.columns if 'emb' in c.lower()]
    if emb_cols:
        print("Cleaning dataset: removing " + str(len(emb_cols)) + " embedding columns")
        sys.stdout.flush()
        df_main = df_main.drop(columns=emb_cols)

    main_cols_to_prefix = ["NOF","NOPF","NOM","NOPM","LOC","WMC","NC","DIT","LCOM","Fan-In","Fan-Out"]
    df_main = apply_prefixes(df_main, main_cols_to_prefix, "class_")

    method_cols_to_prefix = ["cbo","cboModified","fanin","fanout","wmc","rfc","loc"]
    df_methods = apply_prefixes(df_methods, method_cols_to_prefix, "method_")

    class_cols_to_prefix = ["cbo","cboModified","fanin","fanout","wmc","dit","noc","rfc","lcom","lcom*","tcc","lcc", "loc"]
    df_classes = apply_prefixes(df_classes, class_cols_to_prefix, "class_")

    df_main['path_key'] = df_main['File'].apply(normalize_path)
    df_methods['path_key'] = df_methods['file'].apply(normalize_path)
    df_classes['path_key'] = df_classes['file'].apply(normalize_path)

    tracking_info = []
    enriched_method_rows = []

    print("Executing methods enrichment...")
    sys.stdout.flush()
    
    total_rows = len(df_main)
    for idx, row in df_main.iterrows():
        if idx > 0 and idx % 100 == 0:
            sys.stdout.write("\rProcessed {}/{} methods...".format(idx, total_rows))
            sys.stdout.flush()
            
        f_main = row['path_key']
        l_main = row['Line no']
        m_name_main = str(row['Method']).split('(')[0].strip().lower()
        
        file_token = f_main.split('/')[-1]
        
        mask = (df_methods['path_key'].str.endswith(file_token)) & \
               (df_methods['line'] >= l_main - 10) & \
               (df_methods['line'] <= l_main + 10)
        
        match = df_methods[mask].copy()
        status = "no_match"
        
        if match.empty:
            mask_fallback = (df_methods['path_key'].str.endswith(file_token)) & \
                            (df_methods['method'].str.lower().str.contains(m_name_main))
            match = df_methods[mask_fallback].copy()
            if not match.empty: status = "name_fallback"
        else:
            status = "range_match"

        match_row = pd.Series(index=df_methods.columns, dtype='object')
        if not match.empty:
            match['dist'] = (match['line'] - l_main).abs()
            best_match = match.sort_values('dist').iloc[0]
            if status != "name_fallback" and best_match['dist'] == 0: status = "exact"
            match_row = best_match
            
        tracking_info.append({'main_idx': idx, 'file': row['File'], 'method': row['Method'], 'status': status})
        enriched_method_rows.append(match_row)

    sys.stdout.write("\rProcessed {}/{} methods...\n".format(total_rows, total_rows))
    sys.stdout.flush()

    df_methods_matches = pd.DataFrame(enriched_method_rows).reset_index(drop=True)
    
    cols_to_drop = [c for c in ['file', 'line', 'class', 'method', 'dist', 'constructor', 'path_key'] if c in df_methods_matches.columns]
    df_methods_matches = df_methods_matches.drop(columns=cols_to_drop)
    
    df_combined = pd.concat([df_main, df_methods_matches], axis=1)

    print("Executing classes enrichment...")
    sys.stdout.flush()
    
    df_combined['class_key'] = df_combined['Class'].astype(str).str.strip().str.lower()
    df_classes['class_key'] = df_classes['class'].astype(str).apply(lambda x: x.split('.')[-1].split('$')[0].lower())
    
    df_classes_unique = df_classes.drop_duplicates(subset=['path_key', 'class_key']).copy()
    
    df_final = df_combined.merge(
        df_classes_unique,
        on=['path_key', 'class_key'],
        how='left',
        suffixes=('', '_ck_cls')
    )

    to_remove = [c for c in df_final.columns if '_ck_cls' in c or c in ['file', 'class', 'path_key', 'class_key']]
    df_final = df_final.drop(columns=[c for c in to_remove if c in df_final.columns])

    print("Saving enriched dataset: " + out_matches)
    sys.stdout.flush()
    
    Path(out_matches).parent.mkdir(parents=True, exist_ok=True)
    Path(out_track).parent.mkdir(parents=True, exist_ok=True)
    
    df_final.to_csv(out_matches, index=False)
    pd.DataFrame(tracking_info).to_csv(out_track, index=False)
    print("Success: project processed.")
    sys.stdout.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--main', required=True)
    parser.add_argument('--methods', required=True)
    parser.add_argument('--classes', required=True)
    parser.add_argument('--out_matches', required=True)
    parser.add_argument('--out_track', required=True)
    args = parser.parse_args()

    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    def resolve(p):
        path = Path(p)
        return path if path.is_absolute() else BASE_DIR / path

    enrich_dataset(
        str(resolve(args.main)), 
        str(resolve(args.methods)), 
        str(resolve(args.classes)),
        str(resolve(args.out_matches)), 
        str(resolve(args.out_track))
    )