import pandas as pd
import json
import argparse
import os
from pathlib import Path
from collections import Counter

def analyze_with_live_tracking(designite_csv, ref_miner_json, output_matches_csv, output_tracking_csv):
    # Controllo esistenza file di input
    if not os.path.exists(designite_csv):
        print(f"Errore: File Designite non trovato in {designite_csv}")
        return
    if not os.path.exists(ref_miner_json):
        print(f"Errore: File RefactoringMiner non trovato in {ref_miner_json}")
        return

    # Creazione automatica cartelle di output
    Path(output_matches_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(output_tracking_csv).parent.mkdir(parents=True, exist_ok=True)

    print(f"--- Caricamento dati in corso ---")
    df_smells = pd.read_csv(designite_csv)
    
    # Pulizia nomi colonne e mapping dinamico
    df_smells.columns = df_smells.columns.str.strip()
    
    col_map = {}
    possible_type_cols = ['Type', 'Type Name', 'Class Name', 'Class']
    possible_smell_cols = ['Code Smell', 'Smell', 'Implementation Smell', 'Design Smell']

    for c in possible_type_cols:
        if c in df_smells.columns:
            col_map['type'] = c
            break
    
    for c in possible_smell_cols:
        if c in df_smells.columns:
            col_map['smell'] = c
            break

    if 'type' not in col_map:
        print(f"Errore: Non trovo la colonna della Classe. Colonne: {list(df_smells.columns)}")
        return

    # --- DEDUPLICAZIONE ---
    # Raggruppiamo gli smell per classe per evitare righe identiche in output
    active_smells_map = {}
    for _, row in df_smells.iterrows():
        c_name = str(row[col_map['type']])
        s_name = str(row.get(col_map.get('smell'), 'N/A'))
        if c_name not in active_smells_map:
            active_smells_map[c_name] = set()
        active_smells_map[c_name].add(s_name)

    try:
        with open(ref_miner_json, 'r', encoding='utf-8') as f:
            history = json.load(f)
    except Exception as e:
        print(f"Errore nella lettura del JSON: {e}")
        return

    # Liste per i risultati
    matches = []
    tracking_history = []
    ref_counts = Counter()
    
    # FILTRAGGIO: Solo refactoring strutturali rilevanti
    INTERESTING_REFACTORINGS = [
        "Extract Method", 
        "Move Method", 
        "Extract And Move Method",
        #"Extract Class",
        #"Extract Interface",
        "Move Attribute",
        "Push Down Method",
        "Pull Up Method",
        #Extract Subclass",
        #"Extract Superclass"
    ]

    commits = history.get('commits', [])
    print(f"--- Analisi su {len(active_smells_map)} classi smelly e {len(commits)} commit ---")

    for commit in commits:
        sha = commit.get('repository', commit.get('commitId', 'N/A'))
        refactorings = commit.get('refactorings', [])
        
        for ref in refactorings:
            ref_type = ref['type']
            
            # --- 1. GESTIONE RENAME/MOVE ---
            if ref_type in ["Rename Class", "Move Class"]:
                left_loc = ref.get('leftSideLocations', [{}])[0]
                right_loc = ref.get('rightSideLocations', [{}])[0]
                
                if 'filePath' in left_loc and 'filePath' in right_loc:
                    old_path = left_loc['filePath']
                    new_path = right_loc['filePath']
                    old_name = old_path.split('/')[-1].replace('.java', '')
                    new_name = new_path.split('/')[-1].replace('.java', '')

                    if old_name in active_smells_map:
                        # Spostiamo il set di smell al nuovo nome della classe
                        smells_to_move = active_smells_map.pop(old_name)
                        active_smells_map[new_name] = smells_to_move
                        
                        tracking_history.append({
                            'commit': sha,
                            'type': ref_type,
                            'old_path': old_path,
                            'new_path': new_path,
                            'old_name': old_name,
                            'new_name': new_name,
                            'num_smells': len(smells_to_move)
                        })
                        

            # --- 2. FILTRO E MATCHING ---
            elif ref_type in INTERESTING_REFACTORINGS:
                involved_files = [loc.get('filePath', '').split('/')[-1] for loc in ref.get('leftSideLocations', [])]
                
                for f_name in involved_files:
                    class_name = f_name.replace('.java', '')
                    if class_name in active_smells_map:
                        for s_name in active_smells_map[class_name]:
                            matches.append({
                                'commit_sha': sha,
                                'class_name': class_name,
                                'refactoring': ref_type,
                                'smell': s_name,
                                'desc': ref.get('description', '')
                            })
                            ref_counts[ref_type] += 1
                        

    # Esportazione in CSV con rimozione duplicati finali
    out_m = output_matches_csv if output_matches_csv.endswith('.csv') else output_matches_csv + ".csv"
    out_t = output_tracking_csv if output_tracking_csv.endswith('.csv') else output_tracking_csv + ".csv"
    
    pd.DataFrame(matches).drop_duplicates().to_csv(out_m, index=False)
    pd.DataFrame(tracking_history).drop_duplicates().to_csv(out_t, index=False)
    
    print("\n" + "="*40)
    print("ANALISI COMPLETATA")
    print(f"Match totali: {len(matches)}")
    print(f"Spostamenti tracciati: {len(tracking_history)}")
    print("-"*40)
    print("Statistiche Refactoring rilevanti:")
    for ref, count in ref_counts.items():
        print(f" - {ref}: {count}")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Incrocia dati Designite e RefactoringMiner.')
    parser.add_argument('--designite', required=True)
    parser.add_argument('--refminer', required=True)
    parser.add_argument('--out_matches', default='results/matches.csv')
    parser.add_argument('--out_track', default='results/movements.csv')

    args = parser.parse_args()

    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    
    def resolve(p):
        path = Path(p)
        return path if path.is_absolute() else BASE_DIR / path

    analyze_with_live_tracking(
        str(resolve(args.designite)),
        str(resolve(args.refminer)),
        str(resolve(args.out_matches)),
        str(resolve(args.out_track))
    )