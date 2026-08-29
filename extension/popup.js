function render(result) {
  const statusEl = document.getElementById("status");
  const confEl = document.getElementById("confidence");
  const reasonsEl = document.getElementById("reasons");
  const urlEl = document.getElementById("url");

  reasonsEl.innerHTML = "";
  confEl.textContent = "";
  urlEl.textContent = "";

  if (!result || result.status === "checking") {
    statusEl.className = "unknown";
    statusEl.textContent = "Checking...";
    return;
  }

  if (result.status === "skipped") {
    statusEl.className = "unknown";
    statusEl.textContent = "Nothing to check here";
    urlEl.textContent = result.checkedUrl || "";
    return;
  }

  if (result.status === "error") {
    statusEl.className = "unknown";
    statusEl.textContent = "Unable to check this page";
    confEl.textContent = result.error ? `Reason: ${result.error}` : "";
    urlEl.textContent = result.checkedUrl || "";
    return;
  }

  if (result.verdict === "phishing") {
    statusEl.className = "phishing";
    statusEl.textContent = "\u26A0 Phishing";
  } else {
    statusEl.className = "safe";
    statusEl.textContent = "\u2713 Safe";
  }

  confEl.textContent = `Confidence: ${Math.round(result.confidence * 100)}%`;
  (result.reasons || []).forEach((reason) => {
    const li = document.createElement("li");
    li.textContent = reason;
    reasonsEl.appendChild(li);
  });
  urlEl.textContent = result.checkedUrl || "";
}

async function loadCurrentTabResult() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab) return;
  const key = `verdict_${tab.id}`;
  const stored = await chrome.storage.local.get(key);
  render(stored[key]);
}

document.getElementById("recheck").addEventListener("click", async () => {
  const statusEl = document.getElementById("status");
  statusEl.className = "unknown";
  statusEl.textContent = "Checking...";
  await chrome.runtime.sendMessage({ type: "RECHECK_CURRENT_TAB" });
  setTimeout(loadCurrentTabResult, 800);
});

loadCurrentTabResult();

// Live-update the popup if the background worker finishes a check
// while the popup happens to be open.
chrome.storage.onChanged.addListener((changes, area) => {
  if (area !== "local") return;
  chrome.tabs.query({ active: true, currentWindow: true }, ([tab]) => {
    if (!tab) return;
    const key = `verdict_${tab.id}`;
    if (changes[key]) render(changes[key].newValue);
  });
});
