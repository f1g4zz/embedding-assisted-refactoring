import pandas as pd
import gc

def merge_designite_features(df_methods_f, df_smells_f, df_classes_f):
    # 1. Preventive cleanup: remove exact duplicates to optimize the merge
    df_methods_f = df_methods_f.drop_duplicates()
    df_smells_f = df_smells_f.drop_duplicates()
    df_classes_f = df_classes_f.drop_duplicates()

    # 2. Identify common columns for the first merge (Methods + Smells)
    common_cols_1 = list(set(df_methods_f.columns) & set(df_smells_f.columns))
    
    # We use 'inner' to keep only methods that are smelly, but using 'left' 
    # instead of 'inner' is also possible if we want to keep all methods even without smells.
    # For giant projects, 'left' is often more predictable.
    df_merged = pd.merge(
        df_methods_f, 
        df_smells_f, 
        on=common_cols_1, 
        how='inner' 
    )

    # Free memory from DataFrames that are no longer needed
    del df_smells_f
    gc.collect()

    # 3. Second merge (Result + Classes)
    common_cols_2 = list(set(df_merged.columns) & set(df_classes_f.columns))
    
    df_final = pd.merge(
        df_merged, 
        df_classes_f, 
        on=common_cols_2, 
        how='inner'
    )

    # Final cleanup
    del df_merged, df_classes_f
    gc.collect()

    return df_final.drop_duplicates()