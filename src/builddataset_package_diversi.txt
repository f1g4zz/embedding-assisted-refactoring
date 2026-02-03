from load_data import load_arcan_csvs
from features_nodes import build_node_features
from features_smells import build_smell_features
from merge_features import merge_arcan_features
from pathlib import Path
import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

# ===================== CodeBERT embedding =====================
def get_codebert_embedding(code_path, tokenizer, model, device):
    try:
        with open(code_path, "r", encoding="utf-8") as f:
            code = f.read()
    except Exception as e:
        print(f"Errore con {code_path}: {e}")
        return None

    inputs = tokenizer(
        code,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()


def normalize_arcan_path(p: str) -> str:
    """Rimuove i prefissi comuni dei progetti Java per isolare il package/classe"""
    p = p.replace("\\", "/")
    prefixes = ["src/main/java/", "src/test/java/"]
    for prefix in prefixes:
        if p.startswith(prefix):
            return p[len(prefix):]
    return p

def get_package_name(filepath: str) -> str:
    """Estrae il package path rimuovendo il nome del file"""
    path_parts = filepath.replace("\\", "/").split("/")
    return "/".join(path_parts[:-1]) if len(path_parts) > 1 else "root"


# ===================== MAIN =====================
if __name__ == "__main__":

    # ===== Root progetto =====
    BASE_DIR = Path(__file__).resolve().parent.parent.parent

    # Punta alla radice 'src' per includere main e test
    JAVA_SRC_ROOT = (
        BASE_DIR
        / "apache"
        / "commons-lang"
        / "src"
    ).resolve()

    OUTPUT_FILE = "dataset_final_with_packages.csv"

    # ===== Carica ARCAN =====
    component_path = BASE_DIR / "apache/commons-lang/arcanOutput/data/component-metrics.csv"
    smell_path = BASE_DIR / "apache/commons-lang/arcanOutput/data/smell-characteristics.csv"

    df_nodes, df_smells = load_arcan_csvs(component_path, smell_path)

    df_nodes_f = build_node_features(df_nodes, component_type="UNIT")
    df_smells_f = build_smell_features(df_smells)

    df_arcan = (
        merge_arcan_features(df_nodes_f, df_smells_f)
        .drop(columns=["CentralComponent"], errors="ignore")
        .reset_index(drop=True)
    )

    # ===== Setup CodeBERT =====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    model = AutoModel.from_pretrained("microsoft/codebert-base").to(device)
    model.eval()

    # ===== Mappa file reali (Main + Test) =====
    file_map = {}
    print(f"Scansione file in: {JAVA_SRC_ROOT}")
    
    for path in JAVA_SRC_ROOT.rglob("*.java"):
        p_str = path.as_posix()
        # Identifichiamo il percorso relativo al package (dopo java/)
        if "/main/java/" in p_str:
            rel_path = p_str.split("/main/java/")[-1]
        elif "/test/java/" in p_str:
            rel_path = p_str.split("/test/java/")[-1]
        else:
            continue
            
        file_map[rel_path] = str(path)

    print(f"File Java mappati correttamente: {len(file_map)}")

    # ===== Embedding a livello di File =====
    embeddings_list = []
    
    # Aggiungiamo una colonna temporanea per identificare il package
    df_arcan['packageName'] = df_arcan['filePathRelative'].apply(get_package_name)

    print("Generazione embedding per ogni file...")
    for index, row in df_arcan.iterrows():
        arcan_path = normalize_arcan_path(row["filePathRelative"])
        real_path = file_map.get(arcan_path)

        if real_path is None:
            # Se ancora non lo trova, stampa per debug (opzionale)
            # print(f"Salto: {arcan_path}") 
            embeddings_list.append(np.zeros(768))
        else:
            emb = get_codebert_embedding(real_path, tokenizer, model, device)
            embeddings_list.append(emb if emb is not None else np.zeros(768))

    # Creiamo un dataframe temporaneo degli embedding dei file
    file_emb_cols = [f"file_emb_{i}" for i in range(768)]
    df_file_embeddings = pd.DataFrame(embeddings_list, columns=file_emb_cols)
    
    # Uniamo temporaneamente al dataframe Arcan
    df_combined = pd.concat([df_arcan, df_file_embeddings], axis=1)

    # ===== Embedding a livello di Package =====
    print("Calcolo degli embedding medi per package...")
    package_emb_cols = [f"pkg_emb_{i}" for i in range(768)]
    
    # Calcolo della media per ogni gruppo di packageName
    df_pkg_embeddings = df_combined.groupby('packageName')[file_emb_cols].transform('mean')
    df_pkg_embeddings.columns = package_emb_cols

    # ===== Dataset finale =====
    dataset_final = pd.concat([df_combined, df_pkg_embeddings], axis=1)

    dataset_final.to_csv(OUTPUT_FILE, index=False)
    print(f"Dataset finale salvato in {OUTPUT_FILE}")
    print(f"Dimensioni finali: {dataset_final.shape}")