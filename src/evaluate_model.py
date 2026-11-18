
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
    balanced_accuracy_score,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "Features.csv"
ARTIFACTS = ROOT / "artifacts"

config = json.loads(
    (ARTIFACTS / "model_config.json").read_text(encoding="utf-8")
)
features = config["features"]
threshold = config["threshold"]

data = pd.read_csv(DATA_PATH)
X = data[features].astype(np.float32)
y = data["Class_Label"].astype(int)

# Recreate the same stratified split used by train_lstm.py.
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=config["random_state"],
    stratify=y,
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=config["random_state"],
    stratify=y_temp,
)

# Load existing artifacts without modifying them.
model = tf.keras.models.load_model(ARTIFACTS / "insider_lstm.keras")
scaler = joblib.load(ARTIFACTS / "scaler.joblib")

X_test_scaled = scaler.transform(X_test)
X_test_seq = X_test_scaled.reshape(-1, len(features), 1)

scores = model.predict(X_test_seq, verbose=0).ravel()
predictions = (scores >= threshold).astype(int)

print("\n=== CURRENT MODEL: TEST-SET EVALUATION ===")
print("Test records:", len(y_test))
print("Actual class counts:")
print(y_test.value_counts().sort_index().to_string())

print("\nConfusion matrix (rows=actual, columns=predicted):")
print(confusion_matrix(y_test, predictions, labels=[0, 1]))

print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Normal (0)", "Threat (1)"],
        zero_division=0,
    )
)

print("PR-AUC:", round(average_precision_score(y_test, scores), 4))
print("ROC-AUC:", round(roc_auc_score(y_test, scores), 4))
print("Balanced accuracy:", round(
    balanced_accuracy_score(y_test, predictions), 4
))
print("Threshold used:", threshold)
