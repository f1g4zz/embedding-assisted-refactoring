import pandas as pd

def build_smell_features(df_smells):
   
    target_cols = [
        "Project","Package","Class","Method","Smell","Description","File"
    ]
    
    
    df_smells.columns = [c.strip() for c in df_smells.columns]
    

    available_cols = [c for c in target_cols if c in df_smells.columns]
    
    return df_smells[available_cols].copy()