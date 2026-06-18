import argparse
import json
import pandas as pd
from pathlib import Path

def load_methods_snapshot(file_path):

    csv_path = Path(file_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    required_cols = ['Package', 'Class', 'Method']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"File {csv_path} does not contain necessary columns: {required_cols}")

    df['method_key'] = (
        df['Package'].astype(str) + "::" + 
        df['Class'].astype(str) + "::" + 
        df['Method'].astype(str)
    )
    return df

def build_refactoring_map(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    mapping = {}
    
    def extract_class_name(file_path):
        return Path(file_path).stem 

    for commit in data.get('commits', []):
        for ref in commit.get('refactorings', []):
            if ref['type'] in ['Rename Method', 'Move Method', 'Extract Method', 
                               'Pull Up Method', 'Push Down Method', 'Change Method Signature']:
                
                left_locs = ref.get('leftSideLocations', [])
                right_locs = ref.get('rightSideLocations', [])
                
                old = next((loc for loc in left_locs if loc['codeElementType'] == 'METHOD_DECLARATION'), None)
                new = next((loc for loc in right_locs if loc['codeElementType'] == 'METHOD_DECLARATION'), None)
                
                if old and new:
                    old_class = old.get('className', extract_class_name(old['filePath']))
                    new_class = new.get('className', extract_class_name(new['filePath']))
                    
                    old_sig = f"{old_class}::{old['codeElement']}"
                    new_sig = f"{new_class}::{new['codeElement']}"
                    mapping[old_sig] = new_sig
                    
    return mapping

def analyze_evolution(before_file, after_file, refminer_json, output_file):
    print(f"--- Evolution Analysis ---")
    
    df_before = load_methods_snapshot(before_file)
    df_after = load_methods_snapshot(after_file)
    ref_map = build_refactoring_map(refminer_json)
    
    existing_keys_after = set(df_after['method_key'].tolist())
    
    # Identify resolved and persisted code smells:
    # 1. A method smell has 'persisted' if it is still smelly in the 'after' snapshot, or 
    #    if it was refactored (renamed, moved, etc.) and its new signature is still smelly.
    # 2. A method smell is 'resolved' if it no longer exists in the 'after' snapshot and 
    #    has not been tracked to a smelly destination.
    resolved = [] 
    persisted = [] 

    print("matching...")
    for _, row in df_before.iterrows():
        key = row['method_key']
        sig = f"{row['Class']}::{row['Method']}"
        
        if key in existing_keys_after:
            persisted.append(row)
        
        elif sig in ref_map:
            new_sig = ref_map[sig]
            if any(new_sig in k for k in existing_keys_after):
                persisted.append(row)
            else:
                resolved.append(row)
        
        else:
            resolved.append(row)


    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    pd.DataFrame(resolved).to_csv(output_file, index=False)

    print(f"Analysis completed.")
    print(f"- Persisted smells: {len(persisted)}")
    print(f"- Solved smells (resulting rows): {len(resolved)}")
    print(f"File saved in: {output_file}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evolutionary analysis of technical debt.")
    parser.add_argument('--before', required=True, help="Initial CSV snapshot file")
    parser.add_argument('--after', required=True, help="Final CSV snapshot file")
    parser.add_argument('--refminer', required=True, help="JSON file from RefactoringMiner")
    parser.add_argument('--output', required=True, help="Output CSV file")
    
    args = parser.parse_args()

    # Four parents up from project_thesis/src/core/evolution.py to get DesigniteJava/
    BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
    
    def resolve(p):
        path = Path(p)
        return path if path.is_absolute() else BASE_DIR / path

    analyze_evolution(
        resolve(args.before),
        resolve(args.after),
        resolve(args.refminer),
        resolve(args.output)
    )
