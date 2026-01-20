def merge_arcan_features(df_nodes_f, df_smells_f):
    """
    Merge tra:
    - df_nodes_f: features strutturali (nodes.csv)
      → chiave: name
    - df_smells_f: smells (metrics.csv)
      → chiave: CentralComponent
    """

    df = (
        df_nodes_f
        .merge(
            df_smells_f,
            left_on="name",
            right_on="CentralComponent",
            how="left",
            suffixes=("", "_smell")
        )
    )

    # Nodi senza smells → 0
    df.fillna(0, inplace=True)

    return df

