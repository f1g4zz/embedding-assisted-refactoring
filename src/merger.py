import pandas as pd
import argparse
from pathlib import Path


LABELS = [
    "Extract Method", "Extract And Move Method", "Extract Variable",
    "Inline Variable", "Split Variable", "Parameterize Variable",
    "Merge Variable", "Replace Pipeline", "Invert Condition",
    "Merge Conditional Expresion"
]

def merge_labeled_files(input_dir, output_file):
    input_path = Path(input_dir)
    output_path = Path(output_file)
    
    if not input_path.is_dir():
        print(f"Error: {input_dir} is not a valid directory.")
        return

    all_dataframes = []
    
    
    total_removed = 0
    total_kept = 0
    
    csv_files = list(input_path.glob("*.csv"))
    
    if not csv_files:
        print(f"No CSV files found at {input_dir}")
        return

    print(f"Found {len(csv_files)} files.")

    for file in csv_files:
        if file.name == output_path.name:
            continue
            
        df = pd.read_csv(file)
        
        missing_labels = [l for l in LABELS if l not in df.columns]
        if missing_labels:
            print(f"The file {file.name} does not contain the required labels and will be skipped")
            continue


        has_labels_mask = (df[LABELS] != 0).any(axis=1)
        
        num_zeros = (~has_labels_mask).sum() 
        num_with_labels = has_labels_mask.sum() 
        
        total_removed += num_zeros
        total_kept += num_with_labels
        
        filtered_df = df[has_labels_mask]
        
        print(f"File {file.name}: Removed {num_zeros} example where all labels equals 0 {num_with_labels}.")
        
        all_dataframes.append(filtered_df)

    if not all_dataframes:
        print("No valid data to merge.")
        return

    final_df = pd.concat(all_dataframes, ignore_index=True)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    final_df.to_csv(output_path, index=False)
    
    print(f"\n" + "="*40)
    print(f"--- MERGE OP COMPLETED ---")
    print(f"Total discarded examples (all labels equal to 0): {total_removed}")
    print(f"Total examples saved:           {total_kept}")
    print(f"% of mantained examples:      {(total_kept/(total_kept + total_removed)*100):.2f}%")
    print(f"File created at: {output_path}")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CSV Merger: only data with at least a label equaling 0 will be kept.")
    parser.add_argument('--dir', required=True, help='Working directory')
    parser.add_argument('--out', default='merged_dataset_filtered.csv', help='File Name')
    
    args = parser.parse_args()

    # Risoluzione percorsi
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    
    def resolve(p):
        path = Path(p)
        return path if path.is_absolute() else (BASE_DIR / path).resolve()

    merge_labeled_files(resolve(args.dir), resolve(args.out))