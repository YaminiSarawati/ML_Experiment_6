
import json
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

with open(BASE_DIR / "outputs/metrics.json") as f:
    metrics = json.load(f)

plt.figure(figsize=(9, 5))
plt.bar(metrics.keys(), metrics.values())

plt.title("Heart Disease Prediction - Model Performance")
plt.xlabel("Evaluation Metrics")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=20)
plt.tight_layout()

plt.savefig(
    BASE_DIR / "outputs/performance.png"
)

plt.show()

print("Performance graph saved successfully!")
