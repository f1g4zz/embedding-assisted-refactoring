def build_methods_features(df_methods):
    # 1. Pulizia nomi colonne
    df_methods.columns = [c.strip() for c in df_methods.columns]
    
    # 2. Definizione mapping per le metriche
    mapping = {
        "LOC": "method_loc",
        "CC": "method_cc",
        "PC": "method_pc"
    }
    
    # 3. Colonne da tenere: Method + le 3 metriche
    target_cols = ["Class", "Method", "LOC", "CC", "PC"]
    
    # Selezioniamo solo quelle presenti tra le 4 richieste
    available_cols = [c for c in target_cols if c in df_methods.columns]
    
    # 4. Filtro e Rinomina
    df_res = df_methods[available_cols].copy()
    
    # Applichiamo la rinomina solo a LOC, CC, PC (se presenti)
    rename_logic = {k: v for k, v in mapping.items() if k in df_res.columns}
    df_res = df_res.rename(columns=rename_logic)
    
    return df_res