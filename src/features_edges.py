import pandas as pd

def build_edge_features(df_edges):
    # Verifichiamo i nomi reali delle colonne nel CSV per evitare KeyError
    # Cerchiamo 'fromID' o 'fromId' o 'from'
    col_from = [c for c in df_edges.columns if c.lower() == 'fromid'][0]
    col_to = [c for c in df_edges.columns if c.lower() == 'toid'][0]

    # Outgoing edges (Archi in uscita dal nodo)
    out_edges = (
        df_edges
        .groupby(col_from)
        .size()
        .reset_index(name="n_out_edges")
        .rename(columns={col_from: "id"}) # Usiamo 'id' come standard per il merge finale
    )

    # Incoming edges (Archi in entrata nel nodo)
    in_edges = (
        df_edges
        .groupby(col_to)
        .size()
        .reset_index(name="n_in_edges")
        .rename(columns={col_to: "id"}) # Usiamo 'id' come standard
    )

    # Merge delle statistiche degli archi
    df_edges_agg = pd.merge(out_edges, in_edges, on="id", how="outer")
    df_edges_agg.fillna(0, inplace=True)

    return df_edges_agg