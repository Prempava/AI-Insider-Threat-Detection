
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"

config = json.loads(
    (ARTIFACTS / "model_config.json").read_text(encoding="utf-8")
)
features = config["features"]

data = pd.read_csv(ROOT / "Features.csv")
X = data[features].astype(np.float32)
y = data["Class_Label"].astype(int)

# Recreate the original stratified split.
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=config["random_state"],
    stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50,
    random_state=config["random_state"], stratify=y_temp
)

model = tf.keras.models.load_model(ARTIFACTS / "insider_lstm.keras")
scaler = joblib.load(ARTIFACTS / "scaler.joblib")

X_scaled = scaler.transform(X_test)
X_seq = X_scaled.reshape(-1, len(features), 1)
scores = model.predict(X_seq, verbose=0).ravel()
actual = y_test.to_numpy()

print("\n=== THRESHOLD COMPARISON ===")
print("Threshold | Precision | Recall | F1 | False alerts | Missed threats")

for threshold in [0.50, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99]:
    pred = (scores >= threshold).astype(int)

    tp = int(((pred == 1) & (actual == 1)).sum())
    fp = int(((pred == 1) & (actual == 0)).sum())
    fn = int(((pred == 0) & (actual == 1)).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    print(
        f"{threshold:9.2f} | {precision:9.3f} | {recall:6.3f} | "
        f"{f1:.3f} | {fp:12d} | {fn:14d}"
    )

print("\nPR-AUC:", round(average_precision_score(actual, scores), 4))
print("Current configured threshold:", config["threshold"])
print("No model artifacts were modified.")
