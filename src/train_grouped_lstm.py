
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
    balanced_accuracy_score,
)
from imblearn.over_sampling import SMOTE
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "Features.csv"
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

FEATURES = [
    "L1_norm", "L2_norm", "L3_norm", "L4_norm", "L5_norm",
    "L6_norm", "L7_norm", "L8_norm", "L9_norm",
    "E1_norm", "E2_norm", "E3_norm", "E4_norm", "E5_norm",
    "F1_norm", "H1_norm", "H2_norm", "H3_norm",
    "D1_norm", "D2_norm",
]

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# 1. Load data
data = pd.read_csv(DATA_PATH)
X = data[FEATURES].astype(np.float32)
y = data["Class_Label"].astype(int)
groups = data["User"]

if y.isna().any() or groups.isna().any():
    raise ValueError("Labels or employee IDs contain missing values.")

if set(y.unique()) != {0, 1}:
    raise ValueError("Expected binary labels 0 and 1.")

# 2. Split employees: 70% train, 15% validation, 15% test.
split1 = GroupShuffleSplit(
    n_splits=1, test_size=0.30, random_state=SEED
)
train_idx, remaining_idx = next(split1.split(X, y, groups))

remaining_groups = groups.iloc[remaining_idx]
split2 = GroupShuffleSplit(
    n_splits=1, test_size=0.50, random_state=SEED + 1
)
val_rel, test_rel = next(
    split2.split(
        X.iloc[remaining_idx],
        y.iloc[remaining_idx],
        remaining_groups,
    )
)
val_idx = remaining_idx[val_rel]
test_idx = remaining_idx[test_rel]

X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

train_users = set(groups.iloc[train_idx])
val_users = set(groups.iloc[val_idx])
test_users = set(groups.iloc[test_idx])

assert not (train_users & val_users)
assert not (train_users & test_users)
assert not (val_users & test_users)

print("Employees:", len(train_users), len(val_users), len(test_users))
print("Threat records:", int(y_train.sum()), int(y_val.sum()), int(y_test.sum()))
print("Test class counts:\n", y_test.value_counts().sort_index())

if min(int(y_train.sum()), int(y_val.sum()), int(y_test.sum())) == 0:
    raise ValueError(
        "A partition has no threat records. Change the group split seed."
    )

# 3. Fit preprocessing on training employees only.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 4. Balance training data only.
minority_count = int(y_train.sum())
if minority_count < 2:
    raise ValueError("Too few threat records for SMOTE.")

smote = SMOTE(
    random_state=SEED,
    k_neighbors=min(5, minority_count - 1),
)
X_balanced, y_balanced = smote.fit_resample(
    X_train_scaled, y_train
)

# LSTM input: 20 tabular features represented as 20 timesteps.
def as_sequence(values):
    return values.reshape(-1, len(FEATURES), 1)

# 5. Build model.
model = Sequential([
    Input(shape=(len(FEATURES), 1)),
    LSTM(32, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    Dropout(0.4),
    Dense(16, activation="relu"),
    Dense(1, activation="sigmoid"),
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=[tf.keras.metrics.AUC(name="roc_auc")],
)

stop = EarlyStopping(
    monitor="val_loss",
    patience=4,
    restore_best_weights=True,
)

model.fit(
    as_sequence(X_balanced),
    np.asarray(y_balanced),
    validation_data=(as_sequence(X_val_scaled), y_val),
    epochs=20,
    batch_size=256,
    callbacks=[stop],
    verbose=1,
)

# 6. Select threshold using validation employees only.
val_scores = model.predict(as_sequence(X_val_scaled), verbose=0).ravel()
best_threshold = 0.5
best_f1 = -1.0

for threshold in np.arange(0.05, 0.96, 0.01):
    pred = (val_scores >= threshold).astype(int)
    actual = y_val.to_numpy()
    tp = np.sum((pred == 1) & (actual == 1))
    fp = np.sum((pred == 1) & (actual == 0))
    fn = np.sum((pred == 0) & (actual == 1))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = float(threshold)

# 7. Evaluate once on unseen test employees.
test_scores = model.predict(as_sequence(X_test_scaled), verbose=0).ravel()
test_pred = (test_scores >= best_threshold).astype(int)

print("\nSelected validation threshold:", best_threshold)
print("Validation F1:", round(best_f1, 4))
print("\nTest confusion matrix:")
print(confusion_matrix(y_test, test_pred, labels=[0, 1]))
print("\nTest classification report:")
print(classification_report(
    y_test, test_pred,
    labels=[0, 1],
    target_names=["Normal (0)", "Threat (1)"],
    zero_division=0,
))

print("Test PR-AUC:", round(average_precision_score(y_test, test_scores), 4))
print("Test ROC-AUC:", round(roc_auc_score(y_test, test_scores), 4))
print("Test balanced accuracy:", round(
    balanced_accuracy_score(y_test, test_pred), 4
))

# 8. Save under new names; do not overwrite production artifacts.
model.save(ARTIFACTS / "insider_lstm_grouped.keras")
joblib.dump(scaler, ARTIFACTS / "scaler_grouped.joblib")

config = {
    "model_type": "LSTM",
    "features": FEATURES,
    "threshold": best_threshold,
    "input_shape": [len(FEATURES), 1],
    "random_state": SEED,
    "split": "employee_grouped_70_15_15",
    "note": (
        "Experimental model. Employees are separated across partitions. "
        "Features are treated as timesteps, not a temporal activity sequence."
    ),
}
(ARTIFACTS / "model_config_grouped.json").write_text(
    json.dumps(config, indent=2), encoding="utf-8"
)

print("\nSaved experimental artifacts:")
print("artifacts/insider_lstm_grouped.keras")
print("artifacts/scaler_grouped.joblib")
print("artifacts/model_config_grouped.json")
print("Existing production artifacts were not overwritten.")
