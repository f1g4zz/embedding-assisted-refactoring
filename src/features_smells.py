import pandas as pd

def build_smell_features(df_smells):
    # 1. Identifica la colonna ID (vertexId o id)
    id_col = next((c for c in ["vertexId", "id"] if c in df_smells.columns), None)
    
    if id_col is None:
        # Fallback estremo: se Arcan non ha messo l'ID nel CSV degli smell, 
        # dovremo usare CentralComponent, ma gli archi puntano a vertexId.
        print("ATTENZIONE: Nessun ID trovato nel CSV degli Smells!")

    # 2. Seleziona le colonne desiderate
    # Ho aggiunto id_col all'inizio della lista
    cols_to_keep = [id_col, "CentralComponent", "smellType", "Severity", "Size", "Strength"]
    
    # Filtra solo quelle che esistono davvero
    existing_cols = [c for c in cols_to_keep if c is not None and c in df_smells.columns]
    
    # Crea il dataframe con le feature esistenti
    df_features = df_smells[existing_cols].copy()
    
    # Esempio di gestione metriche mancanti (come LOCValue visto nel tuo log)
    if "LOCValue" not in df_features.columns:
        df_features["LOCValue"] = 0
        
    return df_features