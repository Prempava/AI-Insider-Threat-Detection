
const PROTECTED_HOSTS = [
  "web.whatsapp.com",
  "www.linkedin.com",
  "mail.google.com"
];

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === "OPEN_VERIFICATION") {
    const protectedTabId = sender.tab?.id;

    if (protectedTabId === undefined) {
      sendResponse({ success: false });
      return;
    }

    chrome.storage.session.set({
      pendingVerificationTabId: protectedTabId,
      pendingTargetUrl: message.targetUrl
    }).then(() => {
      return chrome.tabs.create({
        url: chrome.runtime.getURL("verify.html")
      });
    }).then(() => {
      sendResponse({ success: true });
    }).catch(() => {
      sendResponse({ success: false });
    });

    return true;
  }

  if (message.type === "VERIFICATION_SUCCESS") {
    chrome.storage.session.get([
      "pendingVerificationTabId"
    ]).then(({ pendingVerificationTabId }) => {
      if (pendingVerificationTabId === undefined) return;

      chrome.tabs.sendMessage(
        pendingVerificationTabId,
        { type: "VERIFICATION_SUCCESS" }
      ).catch(() => {});

      chrome.storage.session.remove([
        "pendingVerificationTabId",
        "pendingTargetUrl"
      ]);
    });

    sendResponse({ received: true });
  }
});
