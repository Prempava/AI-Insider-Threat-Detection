
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
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

# -----------------------------
# 1. Paths and configuration
# -----------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "Features.csv"
ARTIFACT_DIR = ROOT / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

FEATURES = [
    "L1_norm", "L2_norm", "L3_norm", "L4_norm", "L5_norm",
    "L6_norm", "L7_norm", "L8_norm", "L9_norm",
    "E1_norm", "E2_norm", "E3_norm", "E4_norm", "E5_norm",
    "F1_norm", "H1_norm", "H2_norm", "H3_norm",
    "D1_norm", "D2_norm",
]

# -----------------------------
# 2. Load and validate dataset
# -----------------------------
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

data = pd.read_csv(DATA_PATH)

required = FEATURES + ["Class_Label"]
missing = sorted(set(required) - set(data.columns))
if missing:
    raise ValueError(f"Missing required columns: {missing}")

if data[required].isna().any().any():
    raise ValueError("Features or labels contain missing values.")

X = data[FEATURES].astype(np.float32)
y = data["Class_Label"].astype(int)

if set(y.unique()) != {0, 1}:
    raise ValueError("Class_Label must contain both 0 and 1.")

print("Dataset shape:", data.shape)
print("Class counts:\n", y.value_counts().sort_index())

# -----------------------------
# 3. Stratified train/validation/test split
# -----------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50,
    random_state=RANDOM_STATE, stratify=y_temp
)

# Fit preprocessing only on training data.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# SMOTE is applied ONLY to the training partition.
smote = SMOTE(random_state=RANDOM_STATE)
X_train_balanced, y_train_balanced = smote.fit_resample(
    X_train_scaled, y_train
)

# LSTM expects: (samples, timesteps, features).
# Here we treat each of the 21 tabular features as one timestep.
X_train_seq = X_train_balanced.reshape(-1, len(FEATURES), 1)
X_val_seq = X_val_scaled.reshape(-1, len(FEATURES), 1)
X_test_seq = X_test_scaled.reshape(-1, len(FEATURES), 1)

print("Training shape after SMOTE:", X_train_seq.shape)
print("Validation shape:", X_val_seq.shape)
print("Test shape:", X_test_seq.shape)

# -----------------------------
# 4. Build and train LSTM
# -----------------------------
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
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.AUC(name="roc_auc"),
        tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
    ],
)

early_stopping = EarlyStopping(
    monitor="val_loss", patience=4, restore_best_weights=True
)

model.fit(
    X_train_seq,
    np.asarray(y_train_balanced),
    validation_data=(X_val_seq, y_val),
    epochs=20,
    batch_size=256,
    callbacks=[early_stopping],
    verbose=1,
)

# -----------------------------
# 5. Choose threshold on validation data
# -----------------------------
val_probs = model.predict(X_val_seq, verbose=0).ravel()

# Select the threshold with the best validation F1 score.
# This is a starting policy, not a universal security threshold.
thresholds = np.arange(0.05, 0.96, 0.01)
best_threshold = 0.5
best_f1 = -1.0

for threshold in thresholds:
    val_pred = (val_probs >= threshold).astype(int)
    tp = np.sum((val_pred == 1) & (y_val.to_numpy() == 1))
    fp = np.sum((val_pred == 1) & (y_val.to_numpy() == 0))
    fn = np.sum((val_pred == 0) & (y_val.to_numpy() == 1))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = float(threshold)

print(f"\nSelected validation threshold: {best_threshold:.2f}")
print(f"Validation F1 at selected threshold: {best_f1:.4f}")

# -----------------------------
# 6. Final evaluation on untouched test data
# -----------------------------
test_probs = model.predict(X_test_seq, verbose=0).ravel()
test_pred = (test_probs >= best_threshold).astype(int)

print("\nConfusion matrix:")
print(confusion_matrix(y_test, test_pred, labels=[0, 1]))

print("\nClassification report:")
print(classification_report(
    y_test, test_pred, labels=[0, 1],
    target_names=["Normal (0)", "Threat (1)"],
    zero_division=0,
))

print("PR-AUC (Average Precision):",
      round(average_precision_score(y_test, test_probs), 4))
print("ROC-AUC:", round(roc_auc_score(y_test, test_probs), 4))
print("Balanced accuracy:",
      round(balanced_accuracy_score(y_test, test_pred), 4))

# -----------------------------
# 7. Save model and preprocessing artifacts
# -----------------------------
model_path = ARTIFACT_DIR / "insider_lstm.keras"
scaler_path = ARTIFACT_DIR / "scaler.joblib"
config_path = ARTIFACT_DIR / "model_config.json"

model.save(model_path)
joblib.dump(scaler, scaler_path)

config = {
    "model_type": "LSTM",
    "features": FEATURES,
    "threshold": best_threshold,
    "input_shape": [len(FEATURES), 1],
    "random_state": RANDOM_STATE,
    "note": (
        "This model treats tabular features as timesteps; "
        "it is not yet a temporal sequence of user activity."
    ),
}

config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

print("\nSaved model:", model_path)
print("Saved scaler:", scaler_path)
print("Saved config:", config_path)
