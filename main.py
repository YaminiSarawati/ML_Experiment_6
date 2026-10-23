
import json
import joblib
import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split

from src.data_loader import load_data, validate_data
from src.feature_engineering import create_features
from src.preprocessing import create_preprocessor
from src.train import train_model
from src.evaluate import evaluate_model

BASE_DIR = Path(__file__).resolve().parent

def main():
    try:
        with open(BASE_DIR / "config/settings.json") as f:
            config = json.load(f)

        (BASE_DIR / "models").mkdir(exist_ok=True)
        (BASE_DIR / "outputs").mkdir(exist_ok=True)

        print("\nSTEP 1: DATA LOADING")
        df = load_data(
            BASE_DIR / config["dataset_path"]
        )

        print("\nSTEP 2: DATA VALIDATION")
        df = validate_data(
            df, config["target_column"]
        )

        print("\nSTEP 3: FEATURE ENGINEERING")
        X, y = create_features(
            df, config["target_column"]
        )

        print("\nSTEP 4: TRAIN TEST SPLIT")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=config["test_size"],
            random_state=config["random_state"],
            stratify=y
        )

        print("\nSTEP 5: PREPROCESSING")
        preprocessor = create_preprocessor(X_train)

        print("\nSTEP 6: MODEL TRAINING")
        model = train_model(
            X_train, y_train, preprocessor, config
        )

        print("\nSTEP 7: MODEL EVALUATION")
        metrics = evaluate_model(
            model, X_test, y_test
        )

        print("\nSTEP 8: SAVING MODEL")
        joblib.dump(
            model,
            BASE_DIR / "models/heart_pipeline.pkl"
        )

        with open(BASE_DIR / "outputs/metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

        print("\nSTEP 9: SAVING PREDICTIONS")
        results = X_test.copy()
        results["Actual"] = y_test
        results["Predicted"] = model.predict(X_test)

        results.to_csv(
            BASE_DIR / "outputs/predictions.csv",
            index=False
        )

        print("\nPIPELINE EXECUTED SUCCESSFULLY!")

    except Exception as error:
        print("Pipeline failed:", error)
        raise

if __name__ == "__main__":
    main()
