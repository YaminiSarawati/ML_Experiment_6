
import pandas as pd

def create_features(df, target):
    # Remove target and unused num column
    columns_to_drop = [target]

    if "num" in df.columns and target != "num":
        columns_to_drop.append("num")

    X = df.drop(columns=columns_to_drop).copy()

    # Convert target into binary labels
    y = pd.to_numeric(df[target], errors="raise")
    y = (y > 0).astype(int)

    print("Feature engineering completed:", X.shape)
    print("Target distribution:")
    print(y.value_counts())

    return X, y
