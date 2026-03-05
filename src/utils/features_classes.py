def build_class_features(df_class):
    """
    Mantiene solo le colonne specificate, senza rinominarle.
    Gestisce eventuali spazi bianchi nei nomi delle colonne del CSV.
    """
    
    target_cols = [
        "NOF", "NOPF", "NOM", "NOPM", "LOC", 
        "WMC", "NC", "DIT", "LCOM", "Fan-In", "Fan-Out", 
        "File", "Line no"
    ]
    
    
    df_class.columns = [c.strip() for c in df_class.columns]
    
  
    available_cols = [c for c in target_cols if c in df_class.columns]
    
    return df_class[available_cols].copy()