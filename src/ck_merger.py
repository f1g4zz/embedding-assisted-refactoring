import pandas as pd
import numpy as np
import argparse
import re
import sys
from pathlib import Path

def ultra_simple_clean(text):
    """
    Pulisce chiavi di classi e package.
    Gestisce il caso delle classi anonime ($):
    Esempio: 'org.freeplane.ActivatorImpl$Anonymous2' -> 'orgfreeplaneactivatorimpl'
    """
    if pd.isna(text) or text == 'nan': return ""
    t = str(text).lower().strip()
    
    if t.endswith('.java'): t = t[:-5]
    
    t = t.split('$')[0]
    
    for char in ['.', '/', '\\', '_', '-']:
        t = t.replace(char, '')
    return t

def clean_method_name(text):
    """
    Pulisce il nome del metodo.
    Esempio: 'setHelp/1[java.lang.String]' -> 'sethelp'
    """
    if pd.isna(text) or text == 'nan': return ""
    t = str(text).lower().strip()
    # Taglia allo slash, parentesi o quadra
    t = re.split(r'[/(\[]', t)[0]
    t = re.sub(r'[^a-z0-9]', '', t)
    return t

def apply_prefixes(df, columns_to_prefix, prefix):
    """Rinomina le metriche per distinguerle (es. LOC -> class_LOC)"""
    rename_dict = {col: f"{prefix}{col}" for col in df.columns if col in columns_to_prefix}
    return df.rename(columns=rename_dict)

def enrich_dataset(main_path, methods_path, classes_path, out_matches, out_track):
    print(f"\n--- ELABORAZIONE IN CORSO (Anonymous Classes Support) ---")
    df_main = pd.read_csv(main_path)
    df_methods = pd.read_csv(methods_path)
    df_classes = pd.read_csv(classes_path)

    for df in [df_main, df_methods, df_classes]:
        df.columns = [c.strip() for c in df.columns]

    metrics = ["cbo", "cboModified", "fanin", "fanout", "wmc", "rfc", "loc", 
               "dit", "noc", "lcom", "lcom*", "tcc", "lcc", "returnsQty", 
               "variablesQty", "parametersQty"]
    
    df_methods = apply_prefixes(df_methods, metrics, "method_")
    df_classes = apply_prefixes(df_classes, metrics, "class_")


    df_main['match_key_class'] = (df_main['Package'].astype(str) + df_main['Class'].astype(str)).apply(ultra_simple_clean)
    
    df_methods['match_key_class'] = df_methods['class'].apply(ultra_simple_clean)
    df_classes['match_key_class'] = df_classes['class'].apply(ultra_simple_clean)

    df_main['match_key_method'] = df_main['Method'].apply(clean_method_name)
    df_methods['match_key_method'] = df_methods['method'].apply(clean_method_name)


    df_methods_unique = df_methods.drop_duplicates(subset=['match_key_class', 'match_key_method'], keep='first')
    df_classes_unique = df_classes.drop_duplicates(subset=['match_key_class'], keep='first')

    df_combined = df_main.merge(
        df_methods_unique,
        on=['match_key_class', 'match_key_method'],
        how='left',
        suffixes=('', '_meth_dup')
    )

    df_final = df_combined.merge(
        df_classes_unique,
        on='match_key_class',
        how='left',
        suffixes=('', '_cls_dup')
    )


    df_final = df_final.drop_duplicates(subset=['Package', 'Class', 'Method'], keep='first')

    cols_to_drop = [c for c in df_final.columns if '_dup' in c or 
                    c in ['match_key_class', 'match_key_method', 'file', 'class', 'method']]
    
    df_final = df_final.drop(columns=[c for c in cols_to_drop if c in df_final.columns])
    
    if 'line' in df_final.columns:
        cols = list(df_final.columns)
        cols.insert(cols.index('Method') + 1, cols.pop(cols.index('line')))
        df_final = df_final[cols]

    print("\n" + "="*50)
    print(f"REPORT MATCHING:")
    print(f"Righe totali: {len(df_final)}")
    if 'method_wmc' in df_final.columns:
        print(f"Match Metodi (incl. Anonime): {df_final['method_wmc'].notna().sum()}")
    if 'class_cbo' in df_final.columns:
        print(f"Match Classi: {df_final['class_cbo'].notna().sum()}")
    print("="*50 + "\n")

    df_final.to_csv(out_matches, index=False)
    print(f"Dataset salvato in {out_matches}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--main', required=True)
    parser.add_argument('--methods', required=True)
    parser.add_argument('--classes', required=True)
    parser.add_argument('--out_matches', required=True)
    parser.add_argument('--out_track', required=False)
    args = parser.parse_args()
    
    enrich_dataset(args.main, args.methods, args.classes, args.out_matches, args.out_track)