def build_node_features(df_nodes):
    # 1. Identifichiamo la colonna ID (Arcan usa spesso 'vertexId' o 'id')
    # Cerchiamo quella corretta tra le colonne disponibili
    possible_id_cols = ["vertexId", "id", "ID"]
    id_col = next((c for c in possible_id_cols if c in df_nodes.columns), None)
    
    if id_col is None:
        print("ATTENZIONE: Nessuna colonna ID trovata in df_nodes!")
        # In casi estremi, se non c'è ID, il merge con gli edges fallirà.
    
    # 2. Seleziona feature numeriche
    numeric_cols = [
        "AbstractnessMetric",
        "ComponentAtdIndex",
        "ComponentAtdIndexHours",
        "FanIn",
        "FanOut",
        "InstabilityMetric",
        "LinesOfCode",
        "NumberOfChildren",
        "PageRank",
        "TotalAmountOfChanges"
    ]

    # Verifica quali colonne numeriche esistono effettivamente per evitare errori
    existing_numeric = [c for c in numeric_cols if c in df_nodes.columns]

    # 3. Trasformazione Boolean → int
    if "ChangeHasOccurred" in df_nodes.columns:
        df_nodes["ChangeHasOccurred"] = df_nodes["ChangeHasOccurred"].astype(int)
    if "IsNested" in df_nodes.columns:
        df_nodes["IsNested"] = df_nodes["IsNested"].astype(int)

    # 4. Prepariamo la lista finale delle colonne da restituire
    # AGGIUNGIAMO id_col alla lista per non perderlo!
    cols_to_keep = []
    if id_col: cols_to_keep.append(id_col)
    
    cols_to_keep.extend(["name", "filePathRelative", "ComponentType"]) # Metadati necessari
    
    # Aggiungiamo le feature
    extra_features = ["ChangeHasOccurred", "IsNested"]
    for f in (existing_numeric + extra_features):
        if f in df_nodes.columns:
            cols_to_keep.append(f)

    # Restituiamo il dataframe con le colonne selezionate
    return df_nodes[cols_to_keep].copy()