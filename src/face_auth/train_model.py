
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FACE_DATA_DIR = ROOT / "artifacts" / "face_data"
MODEL_PATH = FACE_DATA_DIR / "face_model.yml"
LABELS_PATH = FACE_DATA_DIR / "face_labels.json"


def train_face_model():
    if not hasattr(cv2, "face"):
        raise RuntimeError(
            "OpenCV face module is unavailable. "
            "Check opencv-contrib-python installation."
        )

    images = []
    labels = []
    employee_ids = []

    for employee_dir in sorted(FACE_DATA_DIR.iterdir()):
        if not employee_dir.is_dir():
            continue

        samples = sorted(employee_dir.glob("sample_*.png"))
        if not samples:
            continue

        label = len(employee_ids)
        employee_ids.append(employee_dir.name)

        for sample_path in samples:
            image = cv2.imread(
                str(sample_path),
                cv2.IMREAD_GRAYSCALE,
            )

            if image is None:
                print(f"Skipping unreadable image: {sample_path.name}")
                continue

            image = cv2.resize(image, (200, 200))
            images.append(image)
            labels.append(label)

    if not images:
        raise RuntimeError(
            "No face samples found. Capture employee samples first."
        )

    if len(employee_ids) < 2:
        print(
            "WARNING: Only one employee is enrolled. "
            "This model cannot yet be meaningfully tested "
            "for distinguishing multiple enrolled identities."
        )

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(
        images,
        np.array(labels, dtype=np.int32),
    )

    FACE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    recognizer.write(str(MODEL_PATH))

    LABELS_PATH.write_text(
        json.dumps(
            {str(index): employee_id
             for index, employee_id in enumerate(employee_ids)},
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Face model training completed.")
    print(f"Samples used: {len(images)}")
    print(f"Employees enrolled: {len(employee_ids)}")
    print(f"Model saved: {MODEL_PATH}")
    print(f"Labels saved: {LABELS_PATH}")


if __name__ == "__main__":
    train_face_model()
