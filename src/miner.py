import pandas as pd
import json
import argparse
import os
from pathlib import Path
from collections import Counter

def analyze_with_live_tracking(designite_csv, ref_miner_json, output_matches_csv, output_tracking_csv):
    if not os.path.exists(designite_csv):
        print(f"Error: File Designite not Found")
        return

    Path(output_matches_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(output_tracking_csv).parent.mkdir(parents=True, exist_ok=True)

    print(f"--- Loading data from Designite ---")
    df_smells = pd.read_csv(designite_csv)
    df_smells.columns = df_smells.columns.str.strip()
    
    col_map = {}
    for c in ['Type', 'Type Name', 'Class Name', 'Class']:
        if c in df_smells.columns: col_map['type'] = c; break
    for c in ['Code Smell', 'Smell', 'Implementation Smell', 'Design Smell']:
        if c in df_smells.columns: col_map['smell'] = c; break

    # Grouping by class
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
        print(f"Error JSON: {e}"); return

    matches = []
    tracking_history = []
    
    # Refactoring list
    INTERESTING = [
        "Extract Method",

        "Extract And Move Method",

        "Extract Variable",

        "Inline Variable",

        "Split Variable",

        "Parameterize Variable",

        "Merge Variable",

        "Replace Pipeline",

        "Invert Condition",

        "Merge Conditional Expresion"
    ]

    commits = history.get('commits', [])
    print(f"--- Analisi su {len(active_smells_map)} classi smelly e {len(commits)} commit ---")

    for commit in commits:
        sha = commit.get('repository', commit.get('commitId', 'N/A'))
        
        for ref in commit.get('refactorings', []):
            ref_type = ref['type']
            
            #TRACKING RENAME/MOVE
            if ref_type in ["Rename Class", "Move Class"]:
                left_loc = ref.get('leftSideLocations', [{}])[0]
                right_loc = ref.get('rightSideLocations', [{}])[0]
                if 'filePath' in left_loc and 'filePath' in right_loc:
                    old_path, new_path = left_loc['filePath'], right_loc['filePath']
                    old_name = old_path.split('/')[-1].replace('.java', '')
                    new_name = new_path.split('/')[-1].replace('.java', '')

                    if old_name in active_smells_map:
                        active_smells_map[new_name] = active_smells_map.pop(old_name)
                        tracking_history.append({
                            'commit': sha, 'type': ref_type,
                            'old_path': old_path, 'new_path': new_path,
                            'old_name': old_name, 'new_name': new_name
                        })
                        

            # MATCHING REFACTORING
            elif ref_type in INTERESTING:
                involved_files = [loc.get('filePath', '').split('/')[-1].replace('.java', '') 
                                 for loc in ref.get('leftSideLocations', [])]
                
                for class_name in involved_files:
                    if class_name in active_smells_map:
                        for s_name in active_smells_map[class_name]:
                            matches.append({
                                'commit_sha': sha,
                                'class_name': class_name,
                                'refactoring': ref_type,
                                'smell': s_name,
                                'desc': ref.get('description', '')
                            })
                        
                        

    # Remove duplicates
    df_matches = pd.DataFrame(matches).drop_duplicates()
    df_tracking = pd.DataFrame(tracking_history).drop_duplicates()

    
    out_m = output_matches_csv if output_matches_csv.endswith('.csv') else output_matches_csv + ".csv"
    out_t = output_tracking_csv if output_tracking_csv.endswith('.csv') else output_tracking_csv + ".csv"
    df_matches.to_csv(out_m, index=False)
    df_tracking.to_csv(out_t, index=False)
    
    # Stats
    final_counts = df_matches['refactoring'].value_counts()

    print("\n" + "="*40)
    print("ANALYSIs COMPLETED")
    print(f"Original rows CSV match: {len(df_matches)}")
    print(f"Tracked movements: {len(df_tracking)}")
    print("-"*40)
    print("Number of Refactoring:")
    for ref, count in final_counts.items():
        print(f" - {ref}: {count}")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--designite', required=True)
    parser.add_argument('--refminer', required=True)
    parser.add_argument('--out_matches', default='results/matches.csv')
    parser.add_argument('--out_track', default='results/movements.csv')
    args = parser.parse_args()

    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    def resolve(p):
        path = Path(p)
        return path if path.is_absolute() else BASE_DIR / path

    analyze_with_live_tracking(str(resolve(args.designite)), str(resolve(args.refminer)), 
                               str(resolve(args.out_matches)), str(resolve(args.out_track)))