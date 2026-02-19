import argparse
from pathlib import Path
import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

# Import dai tuoi moduli custom
from load_data import load_designite_csvs
from features_methods import build_methods_features
from features_smells import build_smell_features
from features_classes import build_class_features
from merge_features import merge_designite_features


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


# ===================== MAIN PIPELINE =====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline Designite-CodeBERT Semplificata")
    parser.add_argument("--project", type=str, default="openmrs/openmrs-core")
    parser.add_argument("--output", type=str, default="dataset_final.csv")
    args = parser.parse_args()

    # 1. Configurazione Percorsi
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    PROJECT_ROOT = (BASE_DIR / args.project).resolve()
    print(f"--- Configurazione ---")
    print(f"Root Progetto: {PROJECT_ROOT}")

    # 2. Caricamento e Merge Dati Designite
    print("Caricamento CSV Designite...")
    method_metrics = BASE_DIR / "analyses" / "MethodMetrics.csv"
    smells = BASE_DIR / "analyses" / "ImplementationSmells.csv"
    class_metrics = BASE_DIR / "analyses" / "TypeMetrics.csv"

    df_methods, df_smells, df_classes = load_designite_csvs(method_metrics, smells, class_metrics)
    
    print("Esecuzione Merge e Feature Engineering...")
    df_methods_f = build_methods_features(df_methods)
    df_smells_f = build_smell_features(df_smells)
    df_classes_f = build_class_features(df_classes)
    df_designite = merge_designite_features(df_methods_f, df_smells_f, df_classes_f)

    # --- FILTRO SMELLTYPE (CORRETTO) ---
    # Cerchiamo la colonna che si chiama esattamente 'Smell' (case-insensitive)
    target_col = None
    for c in df_designite.columns:
        if c.lower() == 'smell':
            target_col = c
            break

    if target_col:
        print(f"Filtraggio sulla colonna: {target_col}")
        # Rimuoviamo eventuali spazi bianchi e filtriamo per 'Complex Method'
        df_designite = df_designite[
            df_designite[target_col].str.strip().str.contains("Complex Method", case=False, na=False)
        ].reset_index(drop=True)
        print(f"Righe dopo il filtro 'Complex Method': {len(df_designite)}")
    else:
        print("ERRORE: Colonna 'Smell' non trovata nel DataFrame!")
        print(f"Colonne disponibili: {df_designite.columns.tolist()}")
        exit(1)

   # --- DEDUPLICAZIONE ---
    print(f"Righe prima della deduplicazione: {len(df_designite)}")
    numeric_cols = df_designite.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c.lower() not in ['id', 'graphid', 'fromid', 'toid']]
    
    df_designite = df_designite.groupby('File', as_index=False).agg({
        **{col: 'mean' for col in numeric_cols},
        **{col: 'first' for col in df_designite.columns if col not in numeric_cols and col != 'File'}
    })
    print(f"Righe uniche post-deduplicazione: {len(df_designite)}")

    # ===================== 3. SCANSIONE PROGETTO OTTIMIZZATA =====================
    # Invece di iterare tutto ogni volta, usiamo la fine del path come chiave
    disk_files_map = {}
    print(f"Scansione file Java in {PROJECT_ROOT}...")
    for path in PROJECT_ROOT.rglob("*.java"):
        full_path = str(path.resolve())
        norm_path = full_path.replace("\\", "/").lower()
        # Salviamo l'ultima parte del path (es: com/user/Main.java)
        parts = norm_path.split('/')
        if len(parts) >= 3:
            key = "/".join(parts[-3:]) # Prende le ultime 3 cartelle + nome file
            disk_files_map[key] = full_path
        disk_files_map[norm_path] = full_path # Anche path completo per sicurezza

    # ===================== 4. INIZIALIZZAZIONE CODEBERT =====================
    print("Inizializzazione CodeBERT...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    model = AutoModel.from_pretrained("microsoft/codebert-base").to(device)
    model.eval()

    # ===================== 5. GENERAZIONE EMBEDDING =====================
    all_embeddings = []
    found_count = 0

    print(f"Inizio elaborazione di {len(df_designite)} righe...")

    for idx, row in df_designite.iterrows():
        raw_path = str(row.get("File", ""))
        designite_path_norm = raw_path.replace("\\", "/").lower()
        
        real_path = None
        
        # PROVA 1: Path diretto
        if Path(raw_path).exists():
            real_path = raw_path
        
        # PROVA 2: Lookup veloce tramite mappa (addio loop for!)
        else:
            parts = designite_path_norm.split('/')
            if len(parts) >= 3:
                key = "/".join(parts[-3:])
                real_path = disk_files_map.get(key)

        # Estrazione
        emb = None
        if real_path:
            emb = get_codebert_embedding(real_path, tokenizer, model, device)
        
        if emb is not None:
            all_embeddings.append(emb)
            found_count += 1
        else:
            all_embeddings.append(np.zeros(768))
            # Stampa solo se non lo trova per non intasare la console
            if idx % 50 == 0: 
                print(f"[-] Campione non trovato ({idx}): {designite_path_norm}")

    # ===================== 6. UNIONE E SALVATAGGIO =====================
    emb_df = pd.DataFrame(all_embeddings, columns=[f'emb_{i}' for i in range(768)])
    df_final = pd.concat([df_designite.reset_index(drop=True), emb_df], axis=1)
    df_final.to_csv(args.output, index=False)
    
    print(f"\n--- COMPLETATO ---")
    print(f"Processati con successo: {found_count}/{len(df_designite)}")