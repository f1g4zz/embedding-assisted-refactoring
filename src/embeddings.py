import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import os


CODE_DIR = "./test"          # cartella contenente i file .java
OUTPUT_FILE = "dataset_final.csv"

# ====== Setup CodeBERT ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base")
model.to(device)
model.eval()


# ====== Funzione per ottenere embedding da codice ======
def get_codebert_embedding(code_path):
    try:
        with open(code_path, "r", encoding="utf-8") as f:
            code = f.read()
    except Exception as e:
        print(f"Errore con {code_path}: {e}")
        return None

    # tokenizza e ottieni embedding
    inputs = tokenizer(code, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    # usa mean pooling su last_hidden_state
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return embedding

# ====== Genera embedding per ogni classe ======
embeddings_list = []
for idx, row in arcan_df.iterrows():
    class_file = os.path.join(CODE_DIR, row["class_name"] + ".java")
    embedding = get_codebert_embedding(class_file)
    if embedding is not None:
        embeddings_list.append(embedding)
    else:
        embeddings_list.append([0.0]*768)  # fallback

# ====== Unisci embedding con dati Arcan ======


embedding_df = pd.DataFrame(embeddings_list)
dataset_final = pd.concat([arcan_df.reset_index(drop=True), embedding_df], axis=1)

# ====== Salva dataset pronto per ML ======
dataset_final.to_csv(OUTPUT_FILE, index=False)
print(f"Dataset finale salvato in {OUTPUT_FILE}")
