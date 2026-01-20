from load_data import load_arcan_csvs
from features_nodes import build_node_features
from features_smells import build_smell_features
from merge_features import merge_arcan_features
from pathlib import Path
import pandas as pd
import torch
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
    """Rende il path Arcan confrontabile con i path reali"""
    p = p.replace("\\", "/")
    if p.startswith("src/main/java/"):
        p = p[len("src/main/java/"):]
    return p


# ===================== MAIN =====================
if __name__ == "__main__":

    # ===== Root progetto =====
    BASE_DIR = Path(__file__).resolve().parent.parent.parent

    JAVA_SRC_DIR = (
        BASE_DIR
        / "apache"
        / "commons-lang"
        / "src"
        / "main"
        / "java"
    ).resolve()

    OUTPUT_FILE = "dataset_final.csv"

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

    # ===== Mappa file reali =====
    file_map = {}
    for path in JAVA_SRC_DIR.rglob("*.java"):
        rel_path = path.relative_to(JAVA_SRC_DIR).as_posix()
        file_map[rel_path] = str(path)

    print(f"File Java trovati: {len(file_map)}")

    # ===== Embedding =====
    embeddings_list = []

    for _, row in df_arcan.iterrows():
        arcan_path = normalize_arcan_path(row["filePathRelative"])
        real_path = file_map.get(arcan_path)

        if real_path is None:
            embeddings_list.append([0.0] * 768)
            continue

        emb = get_codebert_embedding(real_path, tokenizer, model, device)
        embeddings_list.append(emb if emb is not None else [0.0] * 768)

    # ===== Dataset finale =====
    embedding_df = pd.DataFrame(embeddings_list)
    dataset_final = pd.concat([df_arcan, embedding_df], axis=1)

    dataset_final.to_csv(OUTPUT_FILE, index=False)
    print(f"Dataset finale salvato in {OUTPUT_FILE}")
