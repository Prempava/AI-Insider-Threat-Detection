
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
    balanced_accuracy_score,
)
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "Features.csv"

SEED = 42
FEATURES = [
    "L1_norm", "L2_norm", "L3_norm", "L4_norm", "L5_norm",
    "L6_norm", "L7_norm", "L8_norm", "L9_norm",
    "E1_norm", "E2_norm", "E3_norm", "E4_norm", "E5_norm",
    "F1_norm", "H1_norm", "H2_norm", "H3_norm",
    "D1_norm", "D2_norm",
]

data = pd.read_csv(DATA_PATH)
X = data[FEATURES].astype(np.float32)
y = data["Class_Label"].astype(int)
groups = data["User"]

# Keep employees separated across train, validation, and test.
split1 = GroupShuffleSplit(
    n_splits=1, test_size=0.30, random_state=SEED
)
train_idx, remaining_idx = next(split1.split(X, y, groups))

split2 = GroupShuffleSplit(
    n_splits=1, test_size=0.50, random_state=SEED + 1
)
val_rel, test_rel = next(
    split2.split(
        X.iloc[remaining_idx],
        y.iloc[remaining_idx],
        groups.iloc[remaining_idx],
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

assert not train_users & val_users
assert not train_users & test_users
assert not val_users & test_users

print("Employees (train/validation/test):",
      len(train_users), len(val_users), len(test_users))
print("Threat records (train/validation/test):",
      int(y_train.sum()), int(y_val.sum()), int(y_test.sum()))

for name, labels in [("validation", y_val), ("test", y_test)]:
    if labels.nunique() < 2:
        raise ValueError(f"{name} partition contains only one class.")

# No scaling or SMOTE is required for Random Forest.
model = RandomForestClassifier(
    n_estimators=300,
    min_samples_leaf=2,
    class_weight="balanced_subsample",
    random_state=SEED,
    n_jobs=-1,
)
sample_weights = compute_sample_weight("balanced", y_train)
model.fit(X_train, y_train, sample_weight=sample_weights)

# Choose threshold using validation employees only.
val_scores = model.predict_proba(X_val)[:, 1]
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

# Evaluate the selected model and threshold on test employees.
test_scores = model.predict_proba(X_test)[:, 1]
test_pred = (test_scores >= best_threshold).astype(int)

print("\nSelected validation threshold:", best_threshold)
print("Validation F1:", round(best_f1, 4))
print("\nTest confusion matrix (actual rows, predicted columns):")
print(confusion_matrix(y_test, test_pred, labels=[0, 1]))
print("\nTest classification report:")
print(classification_report(
    y_test, test_pred,
    labels=[0, 1],
    target_names=["Normal (0)", "Threat (1)"],
    zero_division=0,
))
print("Test PR-AUC:", round(
    average_precision_score(y_test, test_scores), 4
))
print("Test ROC-AUC:", round(
    roc_auc_score(y_test, test_scores), 4
))
print("Test balanced accuracy:", round(
    balanced_accuracy_score(y_test, test_pred), 4
))

# Show which features the forest relied on.
importance = pd.Series(
    model.feature_importances_, index=FEATURES
).sort_values(ascending=False)

print("\nTop 10 feature importances:")
print(importance.head(10).to_string())

print("\nComparison completed. No existing artifacts were changed.")
