
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"

# Load the trained model, scaler, and configuration
model = tf.keras.models.load_model(ARTIFACTS / "insider_lstm.keras")
scaler = joblib.load(ARTIFACTS / "scaler.joblib")
config = json.loads(
    (ARTIFACTS / "model_config.json").read_text(encoding="utf-8")
)

features = config["features"]
threshold = config["threshold"]

# Use one real dataset row for a smoke test
data = pd.read_csv(ROOT / "Features.csv", nrows=2)
sample = data[features].iloc[[0]].astype(np.float32)

# Apply exactly the same preprocessing and input shape
scaled = scaler.transform(sample)
sequence = scaled.reshape(1, len(features), 1)

probability = float(model.predict(sequence, verbose=0)[0][0])
prediction = int(probability >= threshold)

print("Model loaded successfully.")
print("Features:", len(features))
print(f"Decision threshold: {threshold:.2f}")
print(f"Threat probability score: {probability:.4f}")
print("Predicted class:", "Threat" if prediction == 1 else "Normal")
print("\nSmoke test completed.")
