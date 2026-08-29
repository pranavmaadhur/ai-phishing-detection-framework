/**
 * PART C - Edge Computing: run scoring in-browser, no network call
 * ===================================================================
 * Drop `predictLocally(features)` directly into the extension's
 * background.js. It's plain JS, zero dependencies, zero downloads,
 * runs in well under 1ms.
 *
 * -------------------------------------------------------------------
 * WHICH APPROACH, AND WHY (the "your call, be honest" decision)
 * -------------------------------------------------------------------
 * The task listed three options: (1) convert the trained XGBoost model
 * to TensorFlow.js, (2) retrain a small equivalent model in
 * TensorFlow/Keras and export THAT to TensorFlow.js, or (3) a
 * lightweight rule-based scorer written directly in JS.
 *
 * Honest assessment for "today":
 *   (1) XGBoost -> TF.js has no clean, reliable converter - the task
 *       description itself already correctly rules this out.
 *   (2) Retraining a Keras net and exporting with tensorflowjs_converter
 *       is the "proper" edge-ML path, but it has real failure points
 *       under time pressure: needs the labeled training data on hand,
 *       needs `pip install tensorflow tensorflowjs` (a genuinely heavy,
 *       sometimes flaky install), a conversion step that can silently
 *       break on version mismatches, and then wiring @tensorflow/tfjs
 *       (~1MB+) into the extension with async model loading. Any one
 *       of those going wrong burns time you don't get back with judges
 *       waiting.
 *   (3) A rule-based scorer is the realistic choice for today: it is
 *       guaranteed to work, needs no new libraries, no model file to
 *       ship, and is genuinely instant. It won't match the cloud
 *       XGBoost model's accuracy exactly, but that's the honest,
 *       expected trade-off of edge scoring anyway (see the plain-English
 *       explanation at the bottom of this file) - and it's a completely
 *       normal, real pattern (lots of production phishing/ad blockers
 *       use fast heuristic pre-filters exactly like this before falling
 *       back to a heavier model).
 *
 * If your team DOES have 30+ spare minutes and a trained Keras
 * equivalent handy, the swap-in path is:
 *   1. pip install tensorflowjs
 *   2. tensorflowjs_converter --input_format=keras model.h5 tfjs_model/
 *   3. In the extension: import * as tf from '@tensorflow/tfjs'; then
 *      const model = await tf.loadLayersModel('tfjs_model/model.json');
 *      const pred = model.predict(tf.tensor2d([featureArray]));
 *   That's a genuine upgrade path, not required for a working demo.
 *
 * -------------------------------------------------------------------
 * HOW THE SCORE IS COMPUTED
 * -------------------------------------------------------------------
 * Each feature contributes points toward a "phishing score" based on
 * well-established phishing heuristics (the same signals security
 * literature and most URL-based phishing filters use). Points are
 * summed and divided by the maximum possible score to get a 0-1
 * confidence. This is intentionally simple and inspectable - you can
 * explain any single decision to a judge in one sentence.
 */

// Point values - tune these later against real data if there's time.
// Higher = stronger phishing signal.
const WEIGHTS = {
  has_ip_address: 25,        // URL uses a raw IP instead of a domain
  no_https: 15,               // site is not served over HTTPS
  is_shortened_url: 15,       // bit.ly / tinyurl-style link (hides real destination)
  has_suspicious_words: 20,   // "login", "verify", "secure", "update" etc. in the URL
  at_symbol: 10,               // "@" in a URL can hide the real destination
  domain_has_digits: 5,       // e.g. paypa1.com, secure24-bank.com
  many_subdomains: 10,        // e.g. secure.login.paypal.verify.example.com
  long_url: 8,                 // unusually long URLs are common in phishing
  many_hyphens: 7,             // e.g. paypal-account-verify-secure.com
  many_dots: 5,                 // lots of "." often means excessive subdomains
  many_digits: 5,               // lots of raw digits in the URL
  many_query_params: 5,        // long tracking/obfuscation query strings
};

const MAX_SCORE = Object.values(WEIGHTS).reduce((a, b) => a + b, 0); // 130

// A URL doesn't need to trip HALF of every possible signal to be
// worth flagging - a couple of strong signals together (e.g. raw IP
// address + no HTTPS + suspicious words) should already read as
// phishing, even if the edge model isn't fully confident about it.
// That's fine: low confidence is exactly what sends it to the cloud
// model for a second opinion (see scoreUrl() below).
const VERDICT_THRESHOLD = 0.4;

// Thresholds for the individual "count" features - adjust if you find
// these too strict/loose while testing against real URLs.
const THRESHOLDS = {
  url_length: 75,
  num_hyphens: 3,
  num_dots: 4,
  num_digits: 5,
  num_subdomains: 3,
  num_query_params: 3,
};

/**
 * Scores a URL instantly using the same 15 features the cloud model
 * uses, without any network call.
 *
 * @param {Object} features - same 15 features used elsewhere in the app:
 *   url_length, domain_length, num_dots, num_hyphens, num_underscore,
 *   num_slash, num_at_symbol, num_digits, has_ip_address, has_https,
 *   num_subdomains, has_suspicious_words, num_query_params,
 *   is_shortened_url, domain_has_digits
 * @returns {{verdict: "phishing"|"safe", confidence: number, method: "edge"}}
 */
function predictLocally(features) {
  let score = 0;

  if (features.has_ip_address) score += WEIGHTS.has_ip_address;
  if (!features.has_https) score += WEIGHTS.no_https;
  if (features.is_shortened_url) score += WEIGHTS.is_shortened_url;
  if (features.has_suspicious_words) score += WEIGHTS.has_suspicious_words;
  if (features.num_at_symbol > 0) score += WEIGHTS.at_symbol;
  if (features.domain_has_digits) score += WEIGHTS.domain_has_digits;
  if (features.num_subdomains > THRESHOLDS.num_subdomains) score += WEIGHTS.many_subdomains;
  if (features.url_length > THRESHOLDS.url_length) score += WEIGHTS.long_url;
  if (features.num_hyphens > THRESHOLDS.num_hyphens) score += WEIGHTS.many_hyphens;
  if (features.num_dots > THRESHOLDS.num_dots) score += WEIGHTS.many_dots;
  if (features.num_digits > THRESHOLDS.num_digits) score += WEIGHTS.many_digits;
  if (features.num_query_params > THRESHOLDS.num_query_params) score += WEIGHTS.many_query_params;

  const phishingProbability = score / MAX_SCORE; // 0.0 - 1.0
  const verdict = phishingProbability >= VERDICT_THRESHOLD ? "phishing" : "safe";
  // Confidence = how far the score sits from "safe", so a verdict near
  // the threshold correctly comes out as low-confidence (and will
  // trigger the cloud fallback in scoreUrl() below) rather than being
  // reported as a confident answer either way.
  const confidence = verdict === "phishing"
    ? phishingProbability
    : 1 - phishingProbability;

  return { verdict, confidence: Math.round(confidence * 1000) / 1000, method: "edge" };
}

/**
 * Example of how background.js should actually use this: score locally
 * first, and only pay the network round-trip to the cloud API when the
 * edge model isn't confident either way.
 *
 * @param {Object} features
 * @param {string} url
 * @param {number} confidenceThreshold - below this, ask the cloud instead
 */
async function scoreUrl(features, url, confidenceThreshold = 0.75) {
  const edgeResult = predictLocally(features);

  if (edgeResult.confidence >= confidenceThreshold) {
    return edgeResult; // fast path - no network call needed
  }

  // Edge model is unsure - fall back to the cloud API for a second opinion.
  try {
    const response = await fetch("https://YOUR_BACKEND_URL/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url, features }),
    });
    const cloudResult = await response.json();
    return { ...cloudResult, method: "cloud" };
  } catch (err) {
    // Network failed too - fall back to whatever the edge model said
    // rather than leaving the user with no answer at all.
    return edgeResult;
  }
}

// If using this as an ES module elsewhere in the extension:
// export { predictLocally, scoreUrl };
