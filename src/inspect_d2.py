
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
data = pd.read_csv(ROOT / "Features.csv", usecols=["D2_norm", "Class_Label"])

print("D2_norm overall:")
print(data["D2_norm"].describe())

print("\nD2_norm by class:")
print(
    data.groupby("Class_Label")["D2_norm"]
    .describe()
    .to_string()
)

print("\nUnique values:", data["D2_norm"].nunique())
print("\nMost frequent values:")
print(data["D2_norm"].value_counts().head(10).to_string())

print("\nMissing values:", data["D2_norm"].isna().sum())
print("\nCorrelation with label:")
print(data[["D2_norm", "Class_Label"]].corr().iloc[0, 1])
