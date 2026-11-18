
import json
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
FACE_DATA_DIR = ROOT / "artifacts" / "face_data"
MODEL_PATH = FACE_DATA_DIR / "face_model.yml"
LABELS_PATH = FACE_DATA_DIR / "face_labels.json"
CASCADE_PATH = (
    FACE_DATA_DIR / "haarcascade_frontalface_default.xml"
)

# Prototype setting only. This is NOT a security-calibrated threshold.
LBPH_DISTANCE_THRESHOLD = 70.0


def verify_face():
    if not MODEL_PATH.is_file() or not LABELS_PATH.is_file():
        raise FileNotFoundError(
            "Trained model or employee labels not found. "
            "Train the model first."
        )

    if not CASCADE_PATH.is_file():
        raise FileNotFoundError("Face detector XML was not found.")

    detector = cv2.CascadeClassifier(str(CASCADE_PATH))
    if detector.empty():
        raise RuntimeError("Could not load the face detector.")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(MODEL_PATH))

    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError(
            "Could not open the camera. Check camera permissions."
        )

    result = None

    try:
        print("Look at the camera. Press Q to cancel.")

        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Could not read from the camera.")

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(100, 100),
            )

            message = "No face detected"

            if len(faces) == 1:
                x, y, w, h = faces[0]
                face = gray[y:y + h, x:x + w]
                face = cv2.resize(face, (200, 200))

                label, distance = recognizer.predict(face)
                employee_id = labels.get(str(label))

                if employee_id is not None:
                    candidate_match = (
                        distance <= LBPH_DISTANCE_THRESHOLD
                    )
                    message = (
                        f"Candidate: {employee_id} | "
                        f"Distance: {distance:.1f} | "
                        f"Below prototype threshold: {candidate_match}"
                    )
                    result = {
                        "candidate_employee_id": employee_id,
                        "distance": float(distance),
                        "below_prototype_threshold": candidate_match,
                        "verified_for_login": False,
                    }
                else:
                    message = "Unknown label"

                cv2.rectangle(
                    frame, (x, y), (x + w, y + h), (0, 255, 0), 2
                )

            elif len(faces) > 1:
                message = "Multiple faces detected; try again"
                result = None

            cv2.putText(
                frame,
                message[:90],
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Face Verification Test", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if result is not None and len(faces) == 1:
                # Allow time to read the result; press Q to close.
                pass

    finally:
        camera.release()
        cv2.destroyAllWindows()

    if result is None:
        print("No usable face result was obtained.")
    else:
        print("Prototype comparison result:")
        print(json.dumps(result, indent=2))
        print(
            "This result is for testing only; "
            "it does not authorize login."
        )

    return result


if __name__ == "__main__":
    verify_face()
