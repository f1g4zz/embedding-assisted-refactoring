import pandas as pd
import re
import argparse
from pathlib import Path

LABELS = [
    "Extract Method", "Extract And Move Method", "Extract Variable",
    "Inline Variable", "Split Variable", "Parameterize Variable",
    "Merge Variable", "Replace Pipeline", "Invert Condition",
    "Merge Conditional Expresion"
]

def extract_method_name(row):
    desc = row['desc']
    if pd.isna(desc): return None
    
    # Caso 1: Extract Method - serve il metodo ORIGINALE (quello dopo 'extracted from')
    if "extracted from" in desc:
        match = re.search(r'extracted from .*?(\w+)\s*\(', desc)
        if match: return match.group(1)

    # Caso 2: Invert Condition e altri - spesso il metodo è alla fine dopo 'in method'
    if "in method" in desc:
        match = re.search(r'in method .*?(\w+)\s*\(', desc)
        if match: return match.group(1)

    # Caso 3: Fallback generale - la parola subito prima della prima parentesi aperta
    match = re.search(r'(\w+)\s*\(', desc)
    return match.group(1) if match else None

def label_dataset(designite_p, refminer_p, output_p):
    output_path = Path(output_p)
    df = pd.read_csv(Path(designite_p))
    ref_df = pd.read_csv(Path(refminer_p))

    for label in LABELS:
        df[label] = 0

    # Carichiamo i metodi presenti in Designite per un controllo rapido
    # Creiamo un set di chiavi (classe_minuscola, metodo_minuscolo)
    designite_keys = set()
    for _, row in df.iterrows():
        c = str(row['Class']).split('.')[-1].lower()
        m = str(row['Method']).split('(')[0].strip().lower()
        designite_keys.add((c, m))

    ref_map = {}
    skipped_data = [] # Lista per il debug dei metodi skippati
    
    print("Mappatura refactoring e analisi discrepanze...")
    for _, row in ref_df.iterrows():
        method_name = extract_method_name(row)
        if method_name:
            cls_full = str(row['class_name']).replace('$', '.')
            cls = cls_full.split('.')[-1].strip().lower()
            meth = method_name.strip().lower()
            key = (cls, meth)
            
            if key in designite_keys:
                if key not in ref_map:
                    ref_map[key] = set()
                ref_map[key].add(row['refactoring'])
            else:
                # Salviamo i dettagli di ciò che non è stato trovato
                skipped_data.append({
                    'class_refminer': row['class_name'],
                    'method_extracted': method_name,
                    'refactoring': row['refactoring'],
                    'description': row['desc']
                })

    matches_found = 0
    rows_labeled = 0

    # Applicazione etichette sul DataFrame originale
    for idx, row in df.iterrows():
        d_class = str(row['Class']).replace('$', '.').split('.')[-1].strip().lower()
        d_method = str(row['Method']).split('(')[0].strip().lower()
        key = (d_class, d_method)
        
        if key in ref_map:
            applied_any = False
            for ref_name in ref_map[key]:
                if ref_name in LABELS:
                    df.at[idx, ref_name] = 1
                    matches_found += 1
                    applied_any = True
            if applied_any:
                rows_labeled += 1

    # --- REPORT E SALVATAGGIO ---
    print(f"\n--- REPORT FINALE ---")
    print(f"Metodi univoci in Designite: {len(designite_keys)}")
    print(f"Metodi di RefMiner TROVATI in Designite: {len(ref_map)}")
    print(f"Metodi di RefMiner SCARTATI: {len(skipped_data)}")
    print(f"Totale etichette '1' applicate: {matches_found}")

    # Salvataggio file principale
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    # Salvataggio file degli scartati per ispezione
    skipped_file = output_path.parent / "skipped_methods.csv"
    pd.DataFrame(skipped_data).drop_duplicates().to_csv(skipped_file, index=False)
    
    print(f"\n[OK] Dataset etichettato: {output_path}")
    print(f"[DEBUG] Elenco metodi non trovati salvato in: {skipped_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--designite', required=True)
    parser.add_argument('--refminer', required=True)
    parser.add_argument('--out', default='results/dataset_labeled.csv')
    args = parser.parse_args()
    
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    def resolve(p):
        path = Path(p)
        return path if path.is_absolute() else (BASE_DIR / path).resolve()

    label_dataset(resolve(args.designite), resolve(args.refminer), resolve(args.out))