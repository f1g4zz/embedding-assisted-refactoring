import pandas as pd
import argparse
import sys

def print_progress(msg):
    print(f"[*] {msg}")
    sys.stdout.flush()

def ultra_normalize(text):
    if pd.isna(text) or text == 'nan': return ""
    t = str(text).lower().strip()
    if t.endswith('.java'): t = t[:-5]
    t = t.split('$')[0]
    for char in ['.', '/', '\\', '_', '-']:
        t = t.replace(char, '')
    return t

def enrich_dataset(main_path, methods_path, metadata_path, output_path):
    print_progress("Caricamento file...")

    df_methods = pd.read_csv(methods_path, sep='\s+', header=None)
    emb_cols = [f'emb_{i}' for i in range(len(df_methods.columns) - 1)]
    df_methods.columns = ['ID'] + emb_cols

    df_metadata = pd.read_csv(metadata_path, sep='|', header=None, usecols=[0, 2, 3])
    df_metadata.columns = ['ID', 'meta_raw_class', 'meta_raw_line']

    df_main = pd.read_csv(main_path)

    methods_enriched = pd.merge(df_methods, df_metadata, on='ID')

    print_progress("Normalizzazione e preparazione al matching tollerante (±5 righe)...")
    df_main['match_key'] = (df_main['Package'].astype(str) + df_main['Class'].astype(str)).apply(ultra_normalize)
    methods_enriched['match_key'] = methods_enriched['meta_raw_class'].apply(ultra_normalize)

    df_main['line_num'] = pd.to_numeric(df_main['line'], errors='coerce')
    methods_enriched['line_num'] = pd.to_numeric(methods_enriched['meta_raw_line'], errors='coerce')

    df_main = df_main.dropna(subset=['line_num'])
    methods_enriched = methods_enriched.dropna(subset=['line_num'])

    df_main['line_num'] = df_main['line_num'].astype('int64')
    methods_enriched['line_num'] = methods_enriched['line_num'].astype('int64')

    df_main = df_main.sort_values('line_num')
    methods_enriched = methods_enriched.sort_values('line_num')

    df_final = pd.merge_asof(
        df_main,
        methods_enriched,
        on='line_num',        
        by='match_key',      
        direction='nearest',
        tolerance=5,         
        suffixes=('', '_meta_dup')
    )

    cols_to_drop = ['match_key', 'line_num', 'ID', 'meta_raw_class', 'meta_raw_line']
    df_final = df_final.drop(columns=[c for c in cols_to_drop if c in df_final.columns])

    print("\n" + "="*60)
    print(f"{'REPORT MATCHING (FUZZY ±3)':^60}")
    print("-" * 60)
    matches = df_final[emb_cols[0]].notna().sum()
    print(f"Righe totali: {len(df_main)}")
    print(f"Match trovati: {matches} ({(matches/len(df_main))*100:.2f}%)")
    print("="*60 + "\n")

    print_progress(f"Salvataggio in: {output_path}")
    df_final.to_csv(output_path, index=False)
    print_progress("Completato!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--main', required=True)
    parser.add_argument('--methods', required=True)
    parser.add_argument('--metadata', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    try:
        enrich_dataset(args.main, args.methods, args.metadata, args.output)
    except Exception as e:
        print(f"\n[ERRORE] {e}")
        
        if "Permission denied" in str(e):
            print("SUGGERIMENTO: Verifica che l'output sia un PERCORSO FILE (es. D:\\out.csv) e non una CARTELLA.")
            print("Verifica anche che il file non sia aperto in Excel.")
        sys.exit(1)