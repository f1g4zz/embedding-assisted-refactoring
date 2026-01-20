def build_edge_features(df_edges):
    # Outgoing edges
    out_edges = (
        df_edges
        .groupby("fromId")
        .size()
        .reset_index(name="n_out_edges")
        .rename(columns={"fromId": "vertexId"})
    )

    # Incoming edges
    in_edges = (
        df_edges
        .groupby("toId")
        .size()
        .reset_index(name="n_in_edges")
        .rename(columns={"toId": "vertexId"})
    )

    # Merge
    df_edges_agg = pd.merge(out_edges, in_edges, on="vertexId", how="outer")
    df_edges_agg.fillna(0, inplace=True)

    return df_edges_agg
