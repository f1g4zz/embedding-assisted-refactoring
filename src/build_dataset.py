import argparse
from pathlib import Path
import pandas as pd
import torch
import numpy as np
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# Import dai tuoi moduli custom
from load_data import load_designite_csvs
from features_methods import build_methods_features
from features_smells import build_smell_features
from features_classes import build_class_features
from merge_features import merge_designite_features

# ===================== FUNZIONI CORE =====================

def extract_method_by_line(file_path, method_name, line_no):
    """
    Estrae il codice del metodo partendo dalla riga specifica.
    Risolve il problema degli overload (stesso nome, riga diversa).
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        # Le righe nei file partono da 1, Python parte da 0
        # Prendiamo dalla riga indicata fino alla fine del file
        start_idx = max(0, int(line_no) - 1)
        content_from_line = "".join(lines[start_idx:])
        
        # Regex elastica per trovare l'apertura del metodo
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
    parser.add_argument("--project", type=str, default="openmrs/openmrs-core")
    parser.add_argument("--output", type=str, default="dataset_final.csv")
    args = parser.parse_args()

    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    PROJECT_ROOT = (BASE_DIR / args.project).resolve()
    OUTPUT = (BASE_DIR / args.output).resolve()
    
    print(f"--- Configurazione ---")
    print(f"Root Progetto: {PROJECT_ROOT}")

    # 1. Caricamento e Merge
    print("Caricamento CSV Designite...")
    df_methods, df_smells, df_classes = load_designite_csvs(
        PROJECT_ROOT / "MethodMetrics.csv",
        PROJECT_ROOT / "ImplementationSmells.csv",
        PROJECT_ROOT / "TypeMetrics.csv"
    )
    
    print("Esecuzione Merge e Feature Engineering...")
    df_methods_f = build_methods_features(df_methods)
    df_smells_f = build_smell_features(df_smells)
    df_classes_f = build_class_features(df_classes)
    df_designite = merge_designite_features(df_methods_f, df_smells_f, df_classes_f)

    # Filtraggio Complex Method
    target_col = next((c for c in df_designite.columns if c.lower() == 'smell'), None)
    if target_col:
        print(f"Filtraggio sulla colonna: {target_col}")
        df_designite = df_designite[
            df_designite[target_col].str.strip().str.contains("Complex Method", case=False, na=False)
        ].reset_index(drop=True)
        print(f"Righe dopo il filtro 'Complex Method': {len(df_designite)}")
    else:
        print("ERRORE: Colonna 'Smell' non trovata!"); exit(1)

    # Identifichiamo la colonna Line No
    line_col = next((c for c in df_designite.columns if c.lower() in ['line no', 'line_no']), 'Line no')
    
    # Deduplicazione preventiva (File, Method, Line)
    print(f"Deduplicazione basata su File, Method e {line_col}...")
    df_designite = df_designite.drop_duplicates(subset=['File', 'Method', line_col]).reset_index(drop=True)
    print(f"Righe uniche da processare: {len(df_designite)}")

    # 2. Scansione Progetto
    print(f"Scansione file Java...")
    disk_files_map = {}
    for path in PROJECT_ROOT.rglob("*.java"):
        full_path = str(path.resolve())
        norm_path = full_path.replace("\\", "/").lower()
        parts = norm_path.split('/')
        if len(parts) >= 3:
            disk_files_map["/".join(parts[-3:])] = full_path
        disk_files_map[norm_path] = full_path
        disk_files_map[path.name.lower()] = full_path

    # 3. Inizializzazione CodeBERT
    print("Inizializzazione CodeBERT...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    model = AutoModel.from_pretrained("microsoft/codebert-base").to(device)
    model.eval()

    # 4. Generazione Embedding
    all_embeddings = []
    found_c, found_m = 0, 0

    print(f"Inizio elaborazione di {len(df_designite)} righe...")

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
            found_c += 1
            try:
                with open(real_path, "r", encoding="utf-8") as f:
                    full_content = f.read()
                
                c_emb = get_codebert_embedding(full_content, tokenizer, model, device)
                m_code = extract_method_by_line(real_path, method_name, line_no)
                
                # --- AGGIUNTA LOG DEBUG ---
                if m_code:
                    # Stampa anteprima del codice estratto (prime 100 char)
                    clean_preview = m_code.replace('\n', ' ')[:100]
                    #tqdm.write(f"[OK] Metodo: {method_name} (Riga {line_no}) -> Codice: {clean_preview}...")
                    m_emb = get_codebert_embedding(m_code, tokenizer, model, device)
                    found_m += 1
                else:
                    tqdm.write(f"[FALLITO] Metodo non trovato: {method_name} in {real_path} (Riga {line_no})")
                # --------------------------
                
            except Exception:
                pass

        all_embeddings.append(np.concatenate([c_emb, m_emb]))

    # 5. Unione e Pulizia Finale Duplicati Embedding
    cols = [f'class_emb_{i}' for i in range(768)] + [f'method_emb_{i}' for i in range(768)]
    emb_df = pd.DataFrame(all_embeddings, columns=cols)
    df_final = pd.concat([df_designite.reset_index(drop=True), emb_df], axis=1)

    # --- DROP DUPLICATI BASATO SULL'EMBEDDING DEL METODO ---
    # Prendiamo le ultime 10 colonne (feature dell'embedding del metodo) per verificare l'identità
    last_10_cols = cols[-10:]
    print("Esecuzione drop finale duplicati (stesso embedding metodo)...")
    before_drop = len(df_final)
    df_final = df_final.drop_duplicates(subset=last_10_cols).reset_index(drop=True)
    after_drop = len(df_final)
    print(f"Rimossi {before_drop - after_drop} duplicati tecnici.")

    # 6. Salvataggio
    df_final.to_csv(OUTPUT, index=False)
    
    print(f"\n--- COMPLETATO ---")
    print(f"Report: Righe finali {len(df_final)}, Metodi estratti {found_m}")
    print(f"File salvato in: {OUTPUT}")