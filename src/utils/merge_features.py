import pandas as pd
def merge_designite_features(df_methods_f, df_smells_f, df_classes_f):
  
    common_cols_1 = list(set(df_methods_f.columns) & set(df_smells_f.columns))
    
    df_merged_methods = pd.merge(
        df_methods_f, 
        df_smells_f, 
        on=common_cols_1, 
        how='inner'
    )
    
   
    common_cols_2 = list(set(df_merged_methods.columns) & set(df_classes_f.columns))
    
    df_final = pd.merge(
        df_merged_methods, 
        df_classes_f, 
        on=common_cols_2, 
        how='inner'
    )


    df_final = df_final.drop_duplicates()
    
    return df_final