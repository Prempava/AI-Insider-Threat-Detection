
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
    balanced_accuracy_score,
)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"

config = json.loads(
    (ARTIFACTS / "model_config.json").read_text(encoding="utf-8")
)
features = config["features"]

data = pd.read_csv(ROOT / "Features.csv")
X = data[features].astype(np.float32)
y = data["Class_Label"].astype(int)
groups = data["User"]

# Reserve approximately 20% of employees for testing.
splitter = GroupShuffleSplit(
    n_splits=1, test_size=0.20, random_state=42
)
train_idx, test_idx = next(splitter.split(X, y, groups=groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
train_users = set(groups.iloc[train_idx])
test_users = set(groups.iloc[test_idx])

print("Training employees:", len(train_users))
print("Testing employees:", len(test_users))
print("Employee overlap:", len(train_users & test_users))
print("Training threat records:", int(y_train.sum()))
print("Testing threat records:", int(y_test.sum()))
print("Testing class counts:")
print(y_test.value_counts().sort_index())

if y_test.nunique() < 2:
    raise ValueError(
        "Test split has only one class. Choose a different employee split."
    )

model = tf.keras.models.load_model(ARTIFACTS / "insider_lstm.keras")
scaler = joblib.load(ARTIFACTS / "scaler.joblib")

X_test_scaled = scaler.transform(X_test)
X_test_seq = X_test_scaled.reshape(-1, len(features), 1)

scores = model.predict(X_test_seq, verbose=0).ravel()
threshold = float(config["threshold"])
predictions = (scores >= threshold).astype(int)

print("\nThreshold:", threshold)
print("\nConfusion matrix (actual rows, predicted columns):")
print(confusion_matrix(y_test, predictions, labels=[0, 1]))

print("\nClassification report:")
print(classification_report(
    y_test,
    predictions,
    labels=[0, 1],
    target_names=["Normal (0)", "Threat (1)"],
    zero_division=0,
))

print("PR-AUC:", round(average_precision_score(y_test, scores), 4))
print("ROC-AUC:", round(roc_auc_score(y_test, scores), 4))
print(
    "Balanced accuracy:",
    round(balanced_accuracy_score(y_test, predictions), 4),
)
print("\nExisting model artifacts were not modified.")
