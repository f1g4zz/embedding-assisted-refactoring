def build_ml_dataset(df, label_column):
    X = df.drop(columns=["vertexId", label_column]).values
    y = df[label_column].values
    return X, y
