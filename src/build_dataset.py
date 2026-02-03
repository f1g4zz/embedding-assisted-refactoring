import argparse
from pathlib import Path
import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

# Import dai tuoi moduli custom
from load_data import load_arcan_csvs
from features_nodes import build_node_features
from features_smells import build_smell_features
from merge_features import merge_arcan_features

# ===================== FUNZIONI CORE =====================

def get_codebert_embedding(code_path, tokenizer, model, device):
    """Estrae l'embedding CodeBERT per un singolo file."""
    try:
        with open(code_path, "r", encoding="utf-8") as f:
            code = f.read()
        inputs = tokenizer(code, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    except Exception as e:
        return None

def get_package_name(filepath: str) -> str:
    """Estrae il percorso della cartella (package) da un file path."""
    if not isinstance(filepath, str) or filepath in ["", ".", "0", "nan"]:
        return "unknown"
    path_parts = filepath.replace("\\", "/").split("/")
    return "/".join(path_parts[:-1]) if len(path_parts) > 1 else "root"

# ===================== MAIN PIPELINE =====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline Arcan-CodeBERT Semplificata")
    parser.add_argument("--project", type=str, default="openmrs/openmrs-core")
    parser.add_argument("--output", type=str, default="dataset_final.csv")
    args = parser.parse_args()

    # 1. Configurazione Percorsi
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    PROJECT_ROOT = (BASE_DIR / args.project).resolve()
    
    print(f"--- Configurazione ---")
    print(f"Root Progetto: {PROJECT_ROOT}")

    # 2. Caricamento e Merge Dati Arcan
    print("Caricamento CSV Arcan...")
    comp_p = PROJECT_ROOT / "arcanOutput" / "arcanOutput" / "data" / "component-metrics.csv"
    smel_p = PROJECT_ROOT / "arcanOutput" / "arcanOutput" / "data" / "smell-characteristics.csv"
    edge_p = PROJECT_ROOT / "arcanOutput" / "arcanOutput" / "data" / "smell-affects.csv"

    df_nodes, df_smells, df_edges = load_arcan_csvs(comp_p, smel_p, edge_p)
    
    print("Esecuzione Merge e Feature Engineering...")
    df_nodes_f = build_node_features(df_nodes)
    df_smells_f = build_smell_features(df_smells)
    df_arcan = merge_arcan_features(df_nodes_f, df_smells_f, df_edges)

    # --- FILTRO CICLI ---
    smell_type_col = next((c for c in df_arcan.columns if 'smellType' in c), None)
    if smell_type_col:
        df_arcan = df_arcan[df_arcan[smell_type_col].str.contains("cycl", case=False, na=False)].reset_index(drop=True)
    
    if len(df_arcan) == 0:
        print("Nessun ciclo trovato. Fine.")
        exit(0)

    # --- NUOVO: DEDUPLICAZIONE INTELLIGENTE ---
    # Uniamo le righe dello stesso file facendo la media delle metriche numeriche
    print(f"Righe prima della deduplicazione: {len(df_arcan)}")
    
    numeric_cols = df_arcan.select_dtypes(include=[np.number]).columns.tolist()
    # Escludiamo eventuali colonne ID che non ha senso mediare
    numeric_cols = [c for c in numeric_cols if c.lower() not in ['id', 'graphid', 'fromid', 'toid']]
    
    # Raggruppiamo per path: media per i numeri, 'first' per il resto (nomi, tipi smell, etc)
    df_arcan = df_arcan.groupby('filePathRelative', as_index=False).agg({
        **{col: 'mean' for col in numeric_cols},
        **{col: 'first' for col in df_arcan.columns if col not in numeric_cols and col != 'filePathRelative'}
    })
    print(f"Righe uniche post-deduplicazione: {len(df_arcan)}")

    # 3. Mappatura File su Disco
    disk_files = {}
    print(f"Scansione file Java in corso...")
    for path in PROJECT_ROOT.rglob("*.java"):
        try:
            rel_path = path.relative_to(PROJECT_ROOT.parent).as_posix().lower()
            disk_files[rel_path] = str(path.resolve())
        except ValueError:
            continue
    
    # 4. Inizializzazione CodeBERT
    print("Inizializzazione CodeBERT...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    model = AutoModel.from_pretrained("microsoft/codebert-base").to(device)
    model.eval()

    # 5. Generazione Embedding File
    embeddings = []
    found_count = 0

    print("Generazione embedding...")
    for _, row in df_arcan.iterrows():
        raw_path = str(row.get("filePathRelative", "")).replace("\\", "/").lower()
        
        # Protezione contro path invalidi (Package/Root)
        if raw_path in [".", "", "nan", "0", "none"] or len(raw_path) < 3:
            embeddings.append(np.zeros(768))
            continue

        arcan_path = raw_path[1:] if raw_path.startswith("/") else raw_path
        real_path = disk_files.get(arcan_path)

        # Fallback contenuto
        if not real_path and ("/" in arcan_path or ".java" in arcan_path):
            for rel_key, abs_path in disk_files.items():
                if arcan_path in rel_key:
                    real_path = abs_path
                    break

        if real_path:
            emb = get_codebert_embedding(real_path, tokenizer, model, device)
            embeddings.append(emb if emb is not None else np.zeros(768))
            if emb is not None: found_count += 1
        else:
            embeddings.append(np.zeros(768))

    print(f"Embedding completati: {found_count} OK su {len(df_arcan)}")

    # 6. Unione e Calcolo Embedding per Package
    file_emb_cols = [f"file_emb_{i}" for i in range(768)]
    df_file_emb = pd.DataFrame(embeddings, columns=file_emb_cols)
    df_combined = pd.concat([df_arcan.reset_index(drop=True), df_file_emb], axis=1)

    print("Calcolo embedding medi per Package...")
    df_combined['packageName'] = df_combined['filePathRelative'].apply(get_package_name)
    
    pkg_emb_cols = [f"pkg_emb_{i}" for i in range(768)]
    # Usiamo transform mean per assegnare la media del package a ogni riga del file
    df_pkg_means = df_combined.groupby('packageName')[file_emb_cols].transform('mean')
    df_pkg_means.columns = pkg_emb_cols

    # 7. Salvataggio Finale
    dataset_final = pd.concat([df_combined, df_pkg_means], axis=1)
    dataset_final.to_csv(args.output, index=False)
    
    print(f"✅ Operazione completata! Dataset salvato in: {args.output}")