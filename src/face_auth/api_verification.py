
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FACE_DATA_DIR = ROOT / "artifacts" / "face_data"
MODEL_PATH = FACE_DATA_DIR / "face_model.yml"
LABELS_PATH = FACE_DATA_DIR / "face_labels.json"
CASCADE_PATH = FACE_DATA_DIR / "haarcascade_frontalface_default.xml"

# Prototype only. This value is not security-calibrated.
LBPH_DISTANCE_THRESHOLD = 86.5


def verify_employee_face(
    employee_id: str,
    image_bytes: bytes,
) -> dict:
    """Compare a submitted image with the claimed employee's face."""

    if not image_bytes:
        return {"matched": False, "reason": "Empty image."}

    if not all(
        path.is_file()
        for path in (MODEL_PATH, LABELS_PATH, CASCADE_PATH)
    ):
        raise FileNotFoundError("Face verification artifacts are missing.")

    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))

    # Require the claimed employee to be present in the trained model.
    if employee_id not in labels.values():
        return {"matched": False, "reason": "Employee is not enrolled in the model."}

    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if frame is None:
        return {"matched": False, "reason": "Invalid image."}

    detector = cv2.CascadeClassifier(str(CASCADE_PATH))
    if detector.empty():
        raise RuntimeError("Could not load the face detector.")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(100, 100),
    )

    if len(faces) != 1:
        return {
            "matched": False,
            "reason": "Exactly one visible face is required.",
        }

    x, y, w, h = faces[0]
    face = cv2.resize(gray[y:y + h, x:x + w], (200, 200))

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(MODEL_PATH))

    predicted_label, distance = recognizer.predict(face)
    predicted_employee = labels.get(str(predicted_label))
    print("\n[FACE DEBUG]")
    print("Claimed Employee:", employee_id)
    print("Predicted Label:", predicted_label)
    print("Predicted Employee:", predicted_employee)
    print("LBPH Distance:", distance)
    print("Allowed Threshold:", LBPH_DISTANCE_THRESHOLD)

    matched = (
        predicted_employee == employee_id
        and distance <= LBPH_DISTANCE_THRESHOLD
    )

    return {
        "matched": bool(matched),
        "reason": "Prototype face comparison completed."
        if matched
        else "Face did not match the claimed employee.",
        "distance": float(distance),
    }
