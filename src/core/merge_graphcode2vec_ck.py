import os
import argparse
import pandas as pd

# Resolve BASE_PATH dynamically
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge GraphCode2Vec embeddings CSV with CK Metrics report CSV using Class FQN and start line."
    )
    parser.add_argument(
        "--ck_file",
        type=str,
        default=os.path.join(BASE_PATH, "ck_merged", "adempiere_merged.csv"),
        help="Path to the input CK metrics CSV file."
    )
    parser.add_argument(
        "--embeddings_file",
        type=str,
        default=os.path.join(BASE_PATH, "graphcode2vec", "graphcode2vec", "source", "adempiere_embeddings_rich.csv"),
        help="Path to the metadata-rich embeddings CSV file."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=os.path.join(BASE_PATH, "ck_graph2vec_merged", "adempiere_embeddings_merged.csv"),
        help="Path to the merged output CSV file."
    )
    parser.add_argument(
        "--tolerance",
        type=int,
        default=0,
        help="Tolerance in lines for aligning the start line (e.g. 5 for a tolerance of +/- 5 lines). Default: 0 (exact match)."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("==================================================================")
    print("     Merge GNN Embedding Vectors with CK Metrics Report   ")
    print("==================================================================")
    
    # Validate input files
    if not os.path.exists(args.ck_file):
        print(f"Error: The CK file {args.ck_file} does not exist.")
        return
        
    if not os.path.exists(args.embeddings_file):
        print(f"Error: The embeddings file {args.embeddings_file} does not exist.")
        print("Ensure that you have successfully generated adempiere_embeddings_rich.csv.")
        return

    # 1. Loading files
    print(f"\n[1/4] Loading...")
    print(f"  -> CK File: {args.ck_file}")
    ck = pd.read_csv(args.ck_file)
    print(f"     Rows loaded: {len(ck)}")
    
    print(f"  -> Embeddings File: {args.embeddings_file}")
    emb = pd.read_csv(args.embeddings_file)
    print(f"     Rows loaded: {len(emb)}")
    
    # 2. Pre-processing and aligning keys
    print(f"\n[2/4] Pre-processing and aligning keys...")
    
    # Build Fully Qualified Name (Package + Class)
    def get_fqn(row):
        pkg = str(row['Package']).strip() if pd.notna(row['Package']) else ''
        cls = str(row['Class']).strip() if pd.notna(row['Class']) else ''
        if pkg:
            return f"{pkg}.{cls}"
        return cls

    ck['class_fqn'] = ck.apply(get_fqn, axis=1)
    
    # Normalize constructors (from class name to '<init>')
    ck['method_normalized'] = ck.apply(
        lambda r: '<init>' if r['Method'] == r['Class'] else r['Method'], 
        axis=1
    )
    
    # 3. Running merge on Class FQN + Start Line (with tolerance option)
    if args.tolerance == 0:
        print(f"\n[3/4] Running merge (exact match key: Class FQN + Start Line)...")
        merged = pd.merge(
            ck, 
            emb, 
            left_on=['class_fqn', 'line'], 
            right_on=['class_name', 'start_line'], 
            how='inner'
        )
    else:
        print(f"\n[3/4] Running fuzzy merge (Class FQN + Method, tolerance of +/- {args.tolerance} lines)...")
        # Merge on class FQN and normalized method name
        temp_merged = pd.merge(
            ck, 
            emb, 
            left_on=['class_fqn', 'method_normalized'], 
            right_on=['class_name', 'method_name'], 
            how='inner'
        )
        # Filter based on absolute difference between CK line and Soot line
        temp_merged['line_diff'] = (temp_merged['line'] - temp_merged['start_line']).abs()
        merged = temp_merged[temp_merged['line_diff'] <= args.tolerance]
        
        # In case of overloaded methods with the same name, keep only the closest match
        # (minimum line_diff) and drop any duplicates to ensure uniqueness.
        merged = merged.sort_values(by='line_diff').drop_duplicates(subset=['class_fqn', 'line'])
        merged = merged.drop(columns=['line_diff'])
    
    # 4. Saving and outputting statistics
    print(f"\n[4/4] Saving merged data...")
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    merged.to_csv(args.output_file, index=False)
    print(f"  -> Merged file successfully saved to: {args.output_file}")
    
    # Final statistics
    matching_rate = (len(merged) / len(ck)) * 100 if len(ck) > 0 else 0
    print("\n================ FINAL MERGE STATISTICS ================")
    print(f"  Total records in original CK report: {len(ck)}")
    print(f"  Total records in generated embeddings: {len(emb)}")
    print(f"  Total matched records:   {len(merged)}")
    print(f"  Successful matching rate:          {matching_rate:.2f}%")
    print("==============================================================")

if __name__ == '__main__':
    main()
