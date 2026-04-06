import argparse
from pathlib import Path
import pandas as pd
import torch
import numpy as np
import re
import os
import sys
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from utils.load_data import load_designite_csvs
from utils.features_methods import build_methods_features
from utils.features_smells import build_smell_features
from utils.features_classes import build_class_features
from utils.merge_features import merge_designite_features

# ===================== CORE FUNCTIONS =====================

def extract_method_by_line(file_path, method_name, line_no):
    """
    Extract methods from files using Designite metadata (Line no).
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        start_idx = max(0, int(line_no) - 1)
        content_from_line = "".join(lines[start_idx:])
        
        pattern = re.escape(method_name.strip()) + r"\s{0,}\([^)]{0,}\)\s{0,}(?:throws\s+[\w\s,]+)?\s{0,}\{"
        
        match = re.search(pattern, content_from_line)
        if match:
            brace_start_in_fragment = content_from_line.find('{', match.start())
            
            brace_count = 1
            for i in range(brace_start_in_fragment + 1, len(content_from_line)):
                if content_from_line[i] == '{':
                    brace_count += 1
                elif content_from_line[i] == '}':
                    brace_count -= 1
                
                if brace_count == 0:
                    return content_from_line[match.start() : i + 1]
        return None
    except Exception:
        return None

def get_codebert_embedding(text, tokenizer, model, device):
    if not text or text.strip() == "":
        return np.zeros(768)
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    except Exception:
        return np.zeros(768)

# ===================== MAIN PIPELINE =====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline Designite-CodeBERT con Line Mapping")
    parser.add_argument("--project", type=str, required=True, help="Percorso del progetto")
    parser.add_argument("--output", type=str, required=True, help="Percorso del file CSV di output")
    parser.add_argument("--no-embeddings", action="store_true", help="Salta la generazione degli embedding CodeBERT")
    args = parser.parse_args()

    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    
    PROJECT_ROOT = Path(args.project).resolve() if os.path.isabs(args.project) else (BASE_DIR / args.project).resolve()
    OUTPUT = Path(args.output).resolve() if os.path.isabs(args.output) else (BASE_DIR / args.output).resolve()
    
    print("--- Configuration ---")
    print("Project Root: " + str(PROJECT_ROOT))
    print("Output File: " + str(OUTPUT))

    # Loading data
    print("Loading CSV Designite...")
    df_methods, df_smells, df_classes = load_designite_csvs(
        PROJECT_ROOT / "MethodMetrics.csv",
        PROJECT_ROOT / "ImplementationSmells.csv",
        PROJECT_ROOT / "TypeMetrics.csv"
    )
    
    print("Executing Merge & Feature Engineering...")
    df_methods_f = build_methods_features(df_methods)
    df_smells_f = build_smell_features(df_smells)
    df_classes_f = build_class_features(df_classes)
    df_designite = merge_designite_features(df_methods_f, df_smells_f, df_classes_f)

    INTERESTING_SMELLS = [
        "Complex Method", "Long Method", "Feature Envy", "Long Parameter List", 
        "Uncommunicative Name", "Complex Conditional", "Brain Method",
        "Magic Number", "Long Identifier", "Duplicate Code", "Deep Inheritance"
    ]

    #target_col = next((c for c in df_designite.columns if c.lower() == 'smell'), None)

    #if target_col:
    #    mask = df_designite[target_col].str.strip().isin(INTERESTING_SMELLS)
    #    df_designite = df_designite[mask].reset_index(drop=True)
    #    print("Rows after smell filtering: " + str(len(df_designite)))
    #else:
    #    print("ERROR: Column 'Smell' not found!"); sys.exit(1)

    line_col = next((c for c in df_designite.columns if c.lower() in ['line no', 'line_no']), 'Line no')
    
    print("Removing duplicates based on File, Method, " + line_col + "...")
    df_designite = df_designite.drop_duplicates(subset=['File', 'Method', line_col]).reset_index(drop=True)

    if not args.no_embeddings:
        print("Scanning Java files for embeddings...")
        disk_files_map = {}
        for path in PROJECT_ROOT.rglob("*.java"):
            full_path = str(path.resolve())
            norm_path = full_path.replace("\\", "/").lower()
            parts = norm_path.split('/')
            if len(parts) >= 3:
                disk_files_map["/".join(parts[-3:])] = full_path
            disk_files_map[norm_path] = full_path
            disk_files_map[path.name.lower()] = full_path

        print("Initializing CodeBERT...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        model = AutoModel.from_pretrained("microsoft/codebert-base").to(device)
        model.eval()

        all_embeddings = []
        found_m = 0

        print("Working on " + str(len(df_designite)) + " rows...")
        for idx, row in tqdm(df_designite.iterrows(), total=len(df_designite)):
            raw_path = str(row.get("File", ""))
            method_name = str(row.get("Method", ""))
            line_no = row.get(line_col) or 0
            
            path_norm = raw_path.replace("\\", "/").lower()
            real_path = None
            parts = path_norm.split('/')
            key_3 = "/".join(parts[-3:]) if len(parts) >= 3 else "none"
            
            if Path(raw_path).exists(): real_path = raw_path
            elif key_3 in disk_files_map: real_path = disk_files_map[key_3]
            elif path_norm in disk_files_map: real_path = disk_files_map[path_norm]
            elif Path(path_norm).name in disk_files_map: real_path = disk_files_map[Path(path_norm).name]

            c_emb = np.zeros(768)
            m_emb = np.zeros(768)

            if real_path:
                try:
                    with open(real_path, "r", encoding="utf-8") as f:
                        full_content = f.read()
                    
                    c_emb = get_codebert_embedding(full_content, tokenizer, model, device)
                    m_code = extract_method_by_line(real_path, method_name, line_no)
                    
                    if m_code:
                        m_emb = get_codebert_embedding(m_code, tokenizer, model, device)
                        found_m += 1
                except Exception:
                    pass

            all_embeddings.append(np.concatenate([c_emb, m_emb]))

        cols = [f'class_emb_{i}' for i in range(768)] + [f'method_emb_{i}' for i in range(768)]
        emb_df = pd.DataFrame(all_embeddings, columns=cols)
        df_final = pd.concat([df_designite.reset_index(drop=True), emb_df], axis=1)
        
        print("Removing leftover duplicates using embedding values...")
        last_10_cols = cols[-10:]
        df_final = df_final.drop_duplicates(subset=last_10_cols).reset_index(drop=True)
    else:
        print("Skipping CodeBERT embeddings stage.")
        df_final = df_designite
        found_m = 0

    df_final.to_csv(OUTPUT, index=False)
    
    print("\n--- DONE ---")
    print("Total rows: " + str(len(df_final)))
    print("Methods extracted: " + str(found_m))
    print("File saved: " + str(OUTPUT))