from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def create_preprocessor(X):
    numeric_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    if not numeric_features:
        raise ValueError("No numerical features found")

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features)
    ])
