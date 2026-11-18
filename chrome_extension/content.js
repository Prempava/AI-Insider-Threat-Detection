
(() => {
  if (document.getElementById("insider-threat-overlay")) return;

  const overlay = document.createElement("div");
  overlay.id = "insider-threat-overlay";

  Object.assign(overlay.style, {
    position: "fixed",
    inset: "0",
    zIndex: "2147483647",
    background: "#111827",
    color: "white",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    fontFamily: "Arial, sans-serif",
    textAlign: "center"
  });

  const panel = document.createElement("div");
  panel.style.maxWidth = "440px";
  panel.style.padding = "30px";

  const title = document.createElement("h2");
  title.textContent = "🔐 Verify your identity";

  const message = document.createElement("p");
  message.textContent =
    "This page remains covered until employee verification succeeds.";

  const button = document.createElement("button");
  button.textContent = "Open verification";
  Object.assign(button.style, {
    padding: "12px 20px",
    cursor: "pointer",
    borderRadius: "8px",
    border: "0"
  });

  const status = document.createElement("p");
  status.textContent = "Verification required.";

  button.addEventListener("click", () => {
    button.disabled = true;
    status.textContent = "Opening verification screen…";

    chrome.runtime.sendMessage(
      { type: "OPEN_VERIFICATION", targetUrl: location.href },
      (response) => {
        if (chrome.runtime.lastError || !response?.success) {
          status.textContent = "Could not open verification. Please retry.";
          button.disabled = false;
          return;
        }

        status.textContent = "Complete verification in the new tab.";
      }
    );
  });

  panel.append(title, message, button, status);
  overlay.appendChild(panel);
  document.documentElement.appendChild(overlay);

  chrome.runtime.onMessage.addListener((message) => {
    if (message.type === "VERIFICATION_SUCCESS") {
      overlay.remove();
    }
  });
})();
