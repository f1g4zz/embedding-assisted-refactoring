import pandas as pd
import numpy as np

def load_designite_csvs(method_path, smells_path, class_path):
    df_method = pd.read_csv(method_path)
    df_smells = pd.read_csv(smells_path)
    df_class = pd.read_csv(class_path)
    return df_method, df_smells, df_class