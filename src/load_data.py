import pandas as pd
import numpy as np
def load_arcan_csvs(nodes_path, smells_path):
    df_nodes = pd.read_csv(nodes_path)
   # df_edges = pd.read_csv(edges_path)
    df_smells = pd.read_csv(smells_path)
    return df_nodes, df_smells #df_edges,
