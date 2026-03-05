def build_methods_features(df_methods):
   
    df_methods.columns = [c.strip() for c in df_methods.columns]
    
    
    mapping = {
        "LOC": "method_loc",
        "CC": "method_cc",
        "PC": "method_pc"
    }
    
   
    target_cols = ["Method", "LOC", "CC", "PC"]
    
  
    available_cols = [c for c in target_cols if c in df_methods.columns]
    
  
    df_res = df_methods[available_cols].copy()
    
    rename_logic = {k: v for k, v in mapping.items() if k in df_res.columns}
    df_res = df_res.rename(columns=rename_logic)
    
    return df_res