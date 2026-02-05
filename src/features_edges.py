import pandas as pd

def build_edge_features(df_edges):
    """
    Esegue il drop di tutte le colonne non necessarie, mantenendo solo le 4 
    colonne fondamentali (ID e Nomi) senza alterare le righe o i nomi.
    """
    # 1. Identifichiamo i nomi esatti delle colonne presenti nel tuo DataFrame
    # Cerchiamo i nomi originali per fromID, toID, from, to
    target_cols = ["from", "fromId", "to", "toId"]
    
    # Creiamo la lista delle colonne da tenere solo se esistono davvero nel DF
    # Questo evita errori se Arcan non ha esportato una delle 4 colonne
    cols_to_keep = [c for c in df_edges.columns if c in target_cols]

    # Se per caso Arcan usa una capitalizzazione diversa (es. fromId), 
    # questo fallback assicura di non perdere le colonne vitali
    if len(cols_to_keep) < 2:
        cols_to_keep = [c for c in df_edges.columns if c.lower() in [t.lower() for t in target_cols]]

    # 2. Restituiamo il DataFrame "pulito" con solo le 4 colonne (o quelle trovate)
    # Manteniamo tutte le righe originali.
    return df_edges[cols_to_keep].copy()