// ============================ CONFIG ============================
// Flip USE_MOCK to false once your teammate gives you the real API URL,
// and put that URL below. Also update "host_permissions" in
// manifest.json to the same domain (needed for the browser to allow
// the cross-origin fetch call).
const USE_MOCK = true;
const API_URL = "https://your-teammates-api.example.com/predict";
// ==================================================================

const TIMEOUT_MS = 6000;

// Only http(s) pages can be checked — chrome://, about:blank, the
// Chrome Web Store, extension pages, etc. are all skipped.
function isCheckableUrl(url) {
  if (!url) return false;
  return url.startsWith("http://") || url.startsWith("https://");
}

// --- Local mock, used until the real API exists ---
// Simulates network delay and applies a few obvious heuristics so you
// can see both "safe" and "phishing" states in the popup while testing.
async function mockPredict(url) {
  await new Promise((resolve) => setTimeout(resolve, 400 + Math.random() * 400));

  const reasons = [];
  let phishy = false;

  if (url.startsWith("http://")) {
    reasons.push("no HTTPS");
    phishy = true;
  }
  if (/\d+\.\d+\.\d+\.\d+/.test(url)) {
    reasons.push("uses IP address");
    phishy = true;
  }
  if (/(login|verify|secure|account).*\.(top|xyz|ru|tk)/i.test(url)) {
    reasons.push("suspicious domain pattern");
    phishy = true;
  }
  if (url.toLowerCase().includes("phish")) {
    reasons.push("known bad keyword in URL");
    phishy = true;
  }

  if (phishy) {
    return {
      verdict: "phishing",
      confidence: Number((0.75 + Math.random() * 0.24).toFixed(2)),
      reasons: reasons.length ? reasons : ["matched phishing pattern"],
    };
  }
  return {
    verdict: "safe",
    confidence: Number((0.85 + Math.random() * 0.14).toFixed(2)),
    reasons: ["HTTPS present", "domain looks normal"],
  };
}

// --- Real API call, matches the contract exactly ---
async function realPredict(url) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), TIMEOUT_MS);

  try {
    const res = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
      signal: controller.signal,
    });

    if (!res.ok) {
      throw new Error(`API returned status ${res.status}`);
    }

    const data = await res.json();
    if (!data || (data.verdict !== "phishing" && data.verdict !== "safe")) {
      throw new Error("API response did not match the expected shape");
    }
    return data;
  } finally {
    clearTimeout(timeoutId);
  }
}

async function checkUrl(url) {
  try {
    const result = USE_MOCK ? await mockPredict(url) : await realPredict(url);
    return { status: "ok", checkedUrl: url, ...result };
  } catch (err) {
    // Covers timeouts (AbortError), network failures, bad status codes,
    // and malformed responses — all become one friendly error state.
    return {
      status: "error",
      checkedUrl: url,
      error: err && err.name === "AbortError" ? "Request timed out" : (err.message || "Request failed"),
    };
  }
}

// Service workers can be killed and restarted by Chrome at any time,
// so we never keep results only in memory — we always write them to
// chrome.storage.local, keyed by tab id, so the popup can read them
// even if the background worker just woke back up.
async function storeResult(tabId, result) {
  await chrome.storage.local.set({ [`verdict_${tabId}`]: result });
}

async function handleTab(tabId, url) {
  if (!isCheckableUrl(url)) {
    await storeResult(tabId, { status: "skipped", checkedUrl: url });
    return;
  }

  // Show a "checking" state right away so the popup doesn't display a
  // stale verdict from the previous page while the new one loads.
  await storeResult(tabId, { status: "checking", checkedUrl: url });

  const result = await checkUrl(url);
  await storeResult(tabId, result);

  if (result.status === "ok" && result.verdict === "phishing") {
    chrome.tabs.sendMessage(tabId, { type: "PHISHING_WARNING", payload: result }).catch(() => {
      // No content script listening (e.g. restricted page) — safe to ignore.
    });
  }
}

// Fires when a tab finishes loading a new URL.
chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (changeInfo.status === "complete" && tab.url) {
    handleTab(tabId, tab.url);
  }
});

// Fires when the user switches to a different tab, so the popup has
// something to show immediately even without a fresh page load.
chrome.tabs.onActivated.addListener(({ tabId }) => {
  chrome.tabs.get(tabId, (tab) => {
    if (chrome.runtime.lastError || !tab) return;
    handleTab(tabId, tab.url);
  });
});

// Lets the popup ask for a fresh check (the "Re-check this page" button).
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === "RECHECK_CURRENT_TAB") {
    chrome.tabs.query({ active: true, currentWindow: true }, ([tab]) => {
      if (tab) {
        handleTab(tab.id, tab.url).then(() => sendResponse({ ok: true }));
      } else {
        sendResponse({ ok: false });
      }
    });
    return true; // keep the message channel open for the async response
  }
});
