import pandas as pd
import gc

def merge_designite_features(df_methods_f, df_smells_f, df_classes_f):
    # 1. Pulizia preventiva: rimuoviamo duplicati esatti che potrebbero appesantire il merge
    df_methods_f = df_methods_f.drop_duplicates()
    df_smells_f = df_smells_f.drop_duplicates()
    df_classes_f = df_classes_f.drop_duplicates()

    # 2. Identifichiamo le colonne comuni per il primo merge (Metodi + Smell)
    common_cols_1 = list(set(df_methods_f.columns) & set(df_smells_f.columns))
    
    # Usiamo 'left' invece di 'inner' se vuoi mantenere tutti i metodi anche senza smell,
    # oppure restiamo su 'inner' ma con cautela. 
    # Per progetti giganti, 'left' è spesso più prevedibile.
    df_merged = pd.merge(
        df_methods_f, 
        df_smells_f, 
        on=common_cols_1, 
        how='inner' 
    )

    # Liberiamo memoria dai DataFrame che non servono più
    del df_smells_f
    gc.collect()

    # 3. Secondo merge (Risultato + Classi)
    common_cols_2 = list(set(df_merged.columns) & set(df_classes_f.columns))
    
    df_final = pd.merge(
        df_merged, 
        df_classes_f, 
        on=common_cols_2, 
        how='inner'
    )

    # Pulizia finale
    del df_merged, df_classes_f
    gc.collect()

    return df_final.drop_duplicates()