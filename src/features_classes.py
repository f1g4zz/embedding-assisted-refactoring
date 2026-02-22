def build_class_features(df_class):
    """
    Mantiene solo le colonne specificate, senza rinominarle.
    Gestisce eventuali spazi bianchi nei nomi delle colonne del CSV.
    """
    # 1. Lista esatta delle colonne che vuoi tenere
    target_cols = [
        "Class", "NOF", "NOPF", "NOM", "NOPM", "LOC", 
        "WMC", "NC", "DIT", "LCOM", "Fan-In", "Fan-Out", 
        "File", "Line no"
    ]
    
    # 2. Pulizia nomi colonne originali (rimuove spazi bianchi tipo "File " -> "File")
    df_class.columns = [c.strip() for c in df_class.columns]
    
    # 3. Seleziona solo le colonne che esistono effettivamente nel DataFrame
    available_cols = [c for c in target_cols if c in df_class.columns]
    
    return df_class[available_cols].copy()