
from sklearn.datasets import fetch_openml
import pandas as pd
from pathlib import Path

# Create data folder
data_folder = Path("data")
data_folder.mkdir(exist_ok=True)

# Import Heart Disease dataset from OpenML
dataset = fetch_openml(
    data_id=43398,
    as_frame=True
)

# Combine features and target
df = dataset.data.copy()
df["num"] = pd.to_numeric(dataset.target)

# Save dataset automatically
file_path = data_folder / "heart_disease.csv"
df.to_csv(file_path, index=False)

print("Dataset imported successfully!")
print("Dataset shape:", df.shape)
print("Dataset saved at:", file_path)
print(df.head())
