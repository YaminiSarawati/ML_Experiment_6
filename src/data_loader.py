
import pandas as pd

def load_data(path):
    df = pd.read_csv(path, na_values=["?", "NA", ""])
    print("Dataset loaded:", df.shape)
    print("Columns:", df.columns.tolist())
    return df

def validate_data(df, target):
    if target not in df.columns:
        raise ValueError(f"Missing target: {target}")

    df = df.drop_duplicates().copy()

    # Convert target values to numeric
    original_target = df[target].copy()
    df[target] = pd.to_numeric(
        df[target], errors="coerce"
    )

    print("Valid numeric target values:",
          df[target].notna().sum())

    if df[target].notna().sum() == 0:
        print("Original target values:")
        print(original_target.value_counts(dropna=False))
        raise ValueError(
            "Target column contains no numeric values. "
            "Check the dataset labels before training."
        )

    # Convert input features to numeric
    for col in df.columns:
        if col != target:
            df[col] = pd.to_numeric(
                df[col], errors="coerce"
            )

    # Remove rows with missing target only
    df = df.dropna(subset=[target])

    print("Dataset after validation:", df.shape)
    print("Data validation completed")

    return df
