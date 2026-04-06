import pandas as pd
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

def merge_labeled_files(input_dir, output_file):
    input_path = Path(input_dir)
    output_path = Path(output_file)
    output_zeros_path = output_path.parent / f"{output_path.stem}_zeros.csv"
    
    if not input_path.is_dir():
        print(f"Error: {input_dir} is not a valid directory.")
        return

    all_labeled_dfs = []
    all_zeros_dfs = []
    
    total_zeros = 0
    total_labeled = 0
    
    csv_files = list(input_path.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found at {input_dir}")
        return

    print(f"Found {len(csv_files)} files.")

    for file in csv_files:
        if file.name == output_path.name or file.name == output_zeros_path.name:
            continue
            
        df = pd.read_csv(file)
        
        missing_labels = [l for l in LABELS if l not in df.columns]
        if missing_labels:
            print(f"Skipping {file.name}: missing required label columns.")
            continue

        has_labels_mask = (df[LABELS] != 0).any(axis=1)
        
        num_zeros = (~has_labels_mask).sum() 
        num_with_labels = has_labels_mask.sum() 
        
        total_zeros += num_zeros
        total_labeled += num_with_labels
        
        all_labeled_dfs.append(df[has_labels_mask])
        all_zeros_dfs.append(df[~has_labels_mask])
        
        print(f"File {file.name}: Labeled: {num_with_labels} | Zeros: {num_zeros}")

    if all_labeled_dfs:
        final_labeled_df = pd.concat(all_labeled_dfs, ignore_index=True)
        
        label_counts = final_labeled_df[LABELS].sum().astype(int)
        

        output_path.parent.mkdir(parents=True, exist_ok=True)
        final_labeled_df.to_csv(output_path, index=False)
        
        print(f"\n" + "="*40)
        print(f"--- MERGE & SPLIT COMPLETED ---")
        print(f"Total labeled rows (kept): {total_labeled}")
        
        print("\nRefactoring Distribution in merged dataset:")
        for label, count in label_counts.items():
            if count > 0:
                print(f"  - {label.ljust(30)}: {count}")
        
        print(f"\nTotal zero rows (moved):   {total_zeros}")
        print(f"Labeled file created at: {output_path}")
    
    if all_zeros_dfs:
        final_zeros_df = pd.concat(all_zeros_dfs, ignore_index=True)
        final_zeros_df.to_csv(output_zeros_path, index=False)
        print(f"Zeros file created at:   {output_zeros_path}")
        print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CSV Merger & Splitter")
    parser.add_argument('--dir', required=True, help='Working directory')
    parser.add_argument('--out', default='merged_dataset_filtered.csv', help='Main output file name')
    
    args = parser.parse_args()
    
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    def resolve(p):
        path = Path(p)
        return path if path.is_absolute() else (BASE_DIR / path).resolve()

    merge_labeled_files(resolve(args.dir), resolve(args.out))