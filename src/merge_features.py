import pandas as pd
def merge_designite_features(df_methods_f, df_smells_f, df_classes_f):
    # ... (tua parte di clean_col rimane invariata) ...

    # --- 2. STEP 1: MERGE METODI + SMELLS ---
    # Invece di on='Method', usiamo le colonne comuni per evitare doppioni di 'File' o 'Class'
    common_cols_1 = list(set(df_methods_f.columns) & set(df_smells_f.columns))
    
    df_merged_methods = pd.merge(
        df_methods_f, 
        df_smells_f, 
        on=common_cols_1, # Usa Method (e File/Class se presenti in entrambi)
        how='inner'
    )
    
    # --- 3. STEP 2: MERGE CON CLASSI ---
    # Anche qui, usiamo le colonne comuni (sicuramente 'File' e forse 'Class')
    common_cols_2 = list(set(df_merged_methods.columns) & set(df_classes_f.columns))
    
    df_final = pd.merge(
        df_merged_methods, 
        df_classes_f, 
        on=common_cols_2, 
        how='inner'
    )

    # --- 4. PULIZIA FINALE (Opzionale) ---
    # Se dopo i merge hai ancora colonne residue con suffissi (es. LOC_class) 
    # perché non erano chiavi di merge ma avevano lo stesso nome
    df_final = df_final.drop_duplicates()
    
    return df_final