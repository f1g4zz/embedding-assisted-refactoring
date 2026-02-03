import pandas as pd

def merge_arcan_features(df_nodes_f, df_smells_f, df_edges):
    # --- 1. NORMALIZZAZIONE TOTALE ---
    # Forza tutto a stringa, rimuove spazi, rimuove .0
    def clean(val):
        return str(val).replace('.0', '').strip()

    # Identifica le colonne ID dinamicamente
    node_id_col = 'vertexId' if 'vertexId' in df_nodes_f.columns else 'id'
    smell_id_col = 'vertexId' if 'vertexId' in df_smells_f.columns else 'id'
    
    # Pulizia
    df_nodes_f[node_id_col] = df_nodes_f[node_id_col].apply(clean)
    df_smells_f[smell_id_col] = df_smells_f[smell_id_col].apply(clean)
    
    # Identifica colonne edges (fromId/toId)
    e_from = 'fromId' if 'fromId' in df_edges.columns else 'from'
    e_to = 'toId' if 'toId' in df_edges.columns else 'to'
    
    df_edges[e_from] = df_edges[e_from].apply(clean)
    df_edges[e_to] = df_edges[e_to].apply(clean)

    # --- 2. MERGE STEP-BY-STEP CON K.O. CHECK ---
    print(f"Smells pre-merge: {len(df_smells_f)}")

    # Step A: Smell + Edges
    # Colleghiamo lo smell (fromId) all'arco
    df_bridge = pd.merge(
        df_smells_f, 
        df_edges, 
        left_on=smell_id_col, 
        right_on=e_from, 
        how='inner'
    )
    print(f"Smell collegati a Edges: {df_bridge[smell_id_col].nunique()}")

    # Step B: Risultato + Nodi
    # Colleghiamo l'arco (toId) al componente (vertexId)
    df_final = pd.merge(
        df_bridge, 
        df_nodes_f, 
        left_on=e_to, 
        right_on=node_id_col, 
        how='inner', 
        suffixes=('_smell', '_node')
    )

    # --- 3. RECOVERY LOGIC PER CYCLIC HIERARCHY ---
    # Se mancano smell, usiamo il CentralComponent come ancora di salvataggio
    mancanti = set(df_smells_f[smell_id_col]) - set(df_final[smell_id_col + '_smell'] if smell_id_col + '_smell' in df_final.columns else [])
    
    if mancanti and 'CentralComponent' in df_smells_f.columns:
        print(f"Tentativo recupero per {len(mancanti)} smell mancanti via CentralComponent...")
        df_miss = df_smells_f[df_smells_f[smell_id_col].isin(mancanti)]
        df_rec = pd.merge(df_miss, df_nodes_f, left_on='CentralComponent', right_on='name', how='inner')
        
        if not df_rec.empty:
            df_final = pd.concat([df_final, df_rec], ignore_index=True).drop_duplicates()
    
    print(f"Risultato finale: {len(df_final)} righe")
    return df_final