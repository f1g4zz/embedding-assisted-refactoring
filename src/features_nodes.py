def build_node_features(df_nodes, component_type="UNIT"):
    # Filtra per granularità
    #df = df_nodes[df_nodes["ComponentType"] == component_type].copy()
    #df = df_nodes.copy()

    # Seleziona feature numeriche
    numeric_cols = [
        "AbstractnessMetric",
        "ComponentAtdIndex",
        "ComponentAtdIndexHours",
        "ComponentType",
        "filePathRelative",
        "FanIn",
        "FanOut",
        "InstabilityMetric",
        "LinesOfCode",
        "NumberOfChildren",
        "PageRank",
        "TotalAmountOfChanges"
    ]

    # Boolean → int
    df_nodes["ChangeHasOccurred"] = df_nodes["ChangeHasOccurred"].astype(int)
    df_nodes["IsNested"] = df_nodes["IsNested"].astype(int)

    features = numeric_cols + ["ChangeHasOccurred", "IsNested"]

    return df_nodes[["name"] + features]
