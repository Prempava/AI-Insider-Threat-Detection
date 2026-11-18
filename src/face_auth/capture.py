
from pathlib import Path
import re

import cv2

ROOT = Path(__file__).resolve().parents[2]
FACE_DATA_DIR = ROOT / "artifacts" / "face_data"
CASCADE_PATH = (
    ROOT / "artifacts" / "face_data" /
    "haarcascade_frontalface_default.xml"
)

def capture_face_samples(
    employee_id: str,
    consent_confirmed: bool = False,
    sample_limit: int = 30,
) -> int:
    """
    Capture face samples for a consenting employee.

    Prototype only: samples are stored locally and must be
    protected as sensitive biometric data.
    """
    if not consent_confirmed:
        raise ValueError("Employee consent is required.")

    if not re.fullmatch(r"[A-Za-z0-9_-]{3,100}", employee_id):
        raise ValueError("Invalid employee ID.")

    if not 10 <= sample_limit <= 100:
        raise ValueError("Sample limit must be between 10 and 100.")

    if not CASCADE_PATH.is_file():
        raise FileNotFoundError("OpenCV face detector was not found.")

    employee_dir = FACE_DATA_DIR / employee_id
    employee_dir.mkdir(parents=True, exist_ok=True)

    detector = cv2.CascadeClassifier(str(CASCADE_PATH))
    if detector.empty():
        raise RuntimeError("Could not load the face detector.")

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError(
            "Could not open the camera. Check camera permissions."
        )

    saved = 0

    try:
        print("Look at the camera. Press Q to cancel.")
        print("Samples will be saved locally for enrollment.")

        while saved < sample_limit:
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

            for (x, y, w, h) in faces:
                cv2.rectangle(
                    frame, (x, y), (x + w, y + h), (0, 255, 0), 2
                )

            cv2.putText(
                frame,
                f"Samples: {saved}/{sample_limit} | Q: cancel",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Employee Face Enrollment", frame)

            key = cv2.waitKey(100) & 0xFF
            if key == ord("q"):
                break

            # Save only when exactly one face is visible.

            # Save only when exactly one face is visible.
            if len(faces) == 1:
                x, y, w, h = faces[0]
                face = gray[y:y + h, x:x + w]
                face = cv2.resize(face, (200, 200))

                filename = employee_dir / f"sample_{saved:03d}.png"

                if cv2.imwrite(str(filename), face):
                    saved += 1
                    print(f"Saved {saved}/{sample_limit}: {filename.name}")
                else:
                    print("ERROR: Could not save image.")

            elif len(faces) == 0:
                print("No face detected. Face the camera with better lighting.", end="\r")
            else:
                print("Multiple faces detected. Keep only one person in frame.", end="\r")


    finally:
        camera.release()
        cv2.destroyAllWindows()

    print(f"Saved {saved} face samples for {employee_id}.")
    return saved
