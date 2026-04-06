import pandas as pd
import re
import argparse
from pathlib import Path

LABELS = [
            "Change Variable Type",
            "Change Parameter Type",
            "Change Return Type",
            "Extract Method",
            "Move Method",
            "Rename Method",
            "Rename Variable",
            "Rename Parameter",
            "Extract Variable",
            "Add Parameter"
]

def extract_method_name(row):
    desc = row['desc']
    if pd.isna(desc): return None
    
    if "extracted from" in desc:
        match = re.search(r'extracted from .*?(\w+)\s*\(', desc)
        if match: return match.group(1)

    if "in method" in desc:
        match = re.search(r'in method .*?(\w+)\s*\(', desc)
        if match: return match.group(1)

    match = re.search(r'(\w+)\s*\(', desc)
    return match.group(1) if match else None

def label_dataset(designite_p, refminer_p, output_p):
    output_path = Path(output_p)
    df = pd.read_csv(Path(designite_p))
    ref_df = pd.read_csv(Path(refminer_p))

    for label in LABELS:
        df[label] = 0

    designite_keys = set()
    for _, row in df.iterrows():
        c = str(row['Class']).split('.')[-1].lower()
        m = str(row['Method']).split('(')[0].strip().lower()
        designite_keys.add((c, m))

    ref_map = {}
    skipped_data = []
    
    print("Mapping Refactorings from RefMiner...")
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
                skipped_data.append({
                    'class_refminer': row['class_name'],
                    'method_extracted': method_name,
                    'refactoring': row['refactoring'],
                    'description': row['desc']
                })

    matches_found = 0
    
    labels_lower = [l.lower() for l in LABELS]
    labels_map = dict(zip(labels_lower, LABELS)) 

    for idx, row in df.iterrows():
        d_class = str(row['Class']).replace('$', '.').split('.')[-1].strip().lower()
        d_method = str(row['Method']).split('(')[0].strip().lower()
        key = (d_class, d_method)
        
        if key in ref_map:
            for ref_name in ref_map[key]:
                ref_name_clean = ref_name.strip().lower()
                if ref_name_clean in labels_lower:
                    column_name = labels_map[ref_name_clean]
                    df.at[idx, column_name] = 1
                    matches_found += 1

    print(f"\n--- REPORT ---")
    print(f"Total rows in dataset: {len(df)}")
    print(f"Unique Methods in Designite: {len(designite_keys)}")
    print(f"RefMiner Methods matched: {len(ref_map)}")
    print(f"Total '1' labels applied: {matches_found}")
    print(f"Refactorings not matched (saved in skipped_methods.csv): {len(skipped_data)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    skipped_file = output_path.parent / "skipped_methods.csv"
    pd.DataFrame(skipped_data).drop_duplicates().to_csv(skipped_file, index=False)
    
    print(f"\n[OK] Dataset labeled saved in: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--designite', required=True, help="Path to MethodMetrics/Smells CSV from Designite")
    parser.add_argument('--refminer', required=True, help="Path to RefactoringMiner CSV")
    parser.add_argument('--out', default='results/dataset_labeled.csv', help="Output CSV path")
    args = parser.parse_args()
    
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    def resolve(p):
        path = Path(p)
        return path if path.is_absolute() else (BASE_DIR / path).resolve()

    label_dataset(resolve(args.designite), resolve(args.refminer), resolve(args.out))