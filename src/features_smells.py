def build_smell_features(df_smells):
    cols = [
        "CentralComponent",
        "ATDI",
        "ATDI_HOURS",
        "ATDI_WEIGHTED",
        "ATDI_WEIGHTED_HOURS",
        "LOCValue",
        "MeanContTreeDepth",
        "MeanContTreeDistance",
        "NumberOfEdges",
        "PageRankWeighted",
        "Severity",
        "Size",
        "Strength",
        "TotalNumberOfChanges"
    ]

    return df_smells[cols].copy()
