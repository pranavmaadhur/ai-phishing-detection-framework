function injectBanner(result) {
  if (document.getElementById("__phish_detector_banner__")) return;

  const banner = document.createElement("div");
  banner.id = "__phish_detector_banner__";
  banner.style.cssText = `
    position: fixed; top: 0; left: 0; right: 0; z-index: 2147483647;
    background: #a11212; color: white; font-family: system-ui, sans-serif;
    font-size: 14px; padding: 10px 16px; display: flex; align-items: center;
    justify-content: space-between; box-shadow: 0 2px 6px rgba(0,0,0,0.3);
  `;

  const confidencePct = Math.round((result.confidence || 0) * 100);
  const reasons = (result.reasons || []).join(", ");

  const text = document.createElement("span");
  text.textContent = `\u26A0 Warning: this page looks like phishing (${confidencePct}% confidence)${
    reasons ? " \u2014 " + reasons : ""
  }`;

  const closeBtn = document.createElement("button");
  closeBtn.textContent = "Dismiss";
  closeBtn.style.cssText =
    "margin-left:12px;background:white;color:#a11212;border:none;border-radius:4px;padding:4px 8px;cursor:pointer;font-size:12px;flex-shrink:0;";
  closeBtn.addEventListener("click", () => banner.remove());

  banner.appendChild(text);
  banner.appendChild(closeBtn);
  document.documentElement.appendChild(banner);
}

chrome.runtime.onMessage.addListener((message) => {
  if (message.type === "PHISHING_WARNING") {
    injectBanner(message.payload);
  }
});
