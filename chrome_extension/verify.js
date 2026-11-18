
const camera = document.getElementById("camera");
const startButton = document.getElementById("start-camera");
const form = document.getElementById("verification-form");
const verifyButton = document.getElementById("verify-button");
const status = document.getElementById("status");

let cameraStream = null;

function stopCamera() {
  if (cameraStream) {
    cameraStream.getTracks().forEach(track => track.stop());
    cameraStream = null;
  }

  camera.srcObject = null;
}


startButton.addEventListener("click", async () => {
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: true,
      audio: false
    });

    camera.srcObject = cameraStream;
    await camera.play();
    status.textContent = "Camera ready. Position your face clearly.";
  } catch (error) {
    status.textContent = "Camera unavailable or permission denied.";
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const employeeId = document.getElementById("employee-id").value.trim();
  const password = document.getElementById("password").value;

  if (!cameraStream || !camera.videoWidth) {
    status.textContent = "Start the camera and wait for the preview first.";
    return;
  }

  verifyButton.disabled = true;
  status.textContent = "Verifying with the security server…";

  try {
    const canvas = document.createElement("canvas");
    canvas.width = camera.videoWidth;
    canvas.height = camera.videoHeight;
    canvas.getContext("2d").drawImage(camera, 0, 0);

    const blob = await new Promise(resolve =>
      canvas.toBlob(resolve, "image/jpeg", 0.9)
    );

    if (!blob) throw new Error("Could not capture an image.");

    const formData = new FormData();
    formData.append("employee_id", employeeId);
    formData.append("password", password);
    formData.append("image", blob, "face.jpg");

    const response = await fetch(
      "http://127.0.0.1:8000/auth/employee/login/face",
      { method: "POST", body: formData }
    );

    const result = await response.json();

    if (!response.ok || !result.access_token) {
      throw new Error(result.detail || "Verification failed.");
    }
stopCamera();

    // Tell the extension that the API accepted the employee login.
    // Do not persist the employee token in extension storage.
    chrome.runtime.sendMessage(
      { type: "VERIFICATION_SUCCESS" },
      () => {
        if (chrome.runtime.lastError) {
          status.textContent =
            "Login succeeded, but the protected tab could not be notified.";
          return;
        }

        status.textContent = "Employee verified. You may continue.";
        form.reset();
      }
    );
  } catch (error) {
    status.textContent = error.message || "Verification failed.";
  } finally {
    verifyButton.disabled = false;
  }
});

window.addEventListener("beforeunload", () => {
  if (cameraStream) {
    cameraStream.getTracks().forEach(track => track.stop());
  }
});
