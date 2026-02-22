import pandas as pd

def build_smell_features(df_smells):
    """
    Mantiene solo le colonne specificate, senza rinominarle.
    Gestisce eventuali spazi bianchi nei nomi delle colonne del CSV.
    """
    # 1. Lista esatta delle colonne che vuoi tenere
    target_cols = [
        "Project","Package","Class","Smell","Description","File"
    ]
    
    # 2. Pulizia nomi colonne originali (rimuove spazi bianchi tipo "File " -> "File")
    df_smells.columns = [c.strip() for c in df_smells.columns]
    
    # 3. Seleziona solo le colonne che esistono effettivamente nel DataFrame
    available_cols = [c for c in target_cols if c in df_smells.columns]
    
    return df_smells[available_cols].copy()