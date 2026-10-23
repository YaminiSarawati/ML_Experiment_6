
import pandas as pd
from pathlib import Path
from src.predict import predict_new_data

BASE_DIR = Path(__file__).resolve().parent

# Load dataset
df = pd.read_csv(
    BASE_DIR / "data/heart_disease.csv",
    na_values=["?", "NA", ""]
)

# Remove target and unused num column
new_data = df.drop(
    columns=["target", "num"],
    errors="ignore"
).head(5)

# Load saved pipeline and predict
predictions = predict_new_data(
    BASE_DIR / "models/heart_pipeline.pkl",
    new_data
)

print("\nHEART DISEASE PREDICTIONS")

for i, prediction in enumerate(predictions, 1):
    result = (
        "Heart Disease"
        if prediction == 1
        else "No Heart Disease"
    )
    print(f"Patient {i}: {result}")

# Save predictions
output = new_data.copy()
output["Prediction"] = predictions

output.to_csv(
    BASE_DIR / "outputs/new_predictions.csv",
    index=False
)

print("\nPredictions saved successfully!")
