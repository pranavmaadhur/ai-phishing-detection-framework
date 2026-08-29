"""
Phishing Detection API — URL, Text, and Image analyzers
==========================================================
POST /predict        -> URL analyzer.   { "verdict": "phishing"|"safe", "confidence": 0.87, "reasons": [...] }
POST /analyze-text    -> Text analyzer.  Same response shape, takes { "text": "..." } instead of a url.
POST /analyze-image   -> Image analyzer. Same response shape, takes an uploaded image file.

HOW THIS FILE IS ORGANIZED (read this first):
1. Imports + app setup + CORS
2. FAKE_MODE switch — flip one variable when your teammate's model.pkl arrives
3. extract_features(url) — turns a URL into the 15 numbers the model expects
4. The fake predictor (used while FAKE_MODE = True)
5. The real predictor (used once FAKE_MODE = False and model.pkl exists)
6. reasons generator — turns feature values into plain-English explanations
7. The /predict endpoint (URL analyzer)
8. The /analyze-text endpoint (Text/message analyzer)
9. The /analyze-image endpoint (Image analyzer)
"""

import re
import pickle
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# 1. App setup + CORS
# ---------------------------------------------------------------------------

app = FastAPI(title="Phishing URL Detection API")

# A Chrome extension calls this API from a "chrome-extension://..." origin,
# which is not a normal website origin. The simplest fix is to allow all
# origins. If you want to lock it down later to just your extension's ID,
# swap allow_origins=["*"] for allow_origins=["chrome-extension://<your-id>"].
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# 2. THE ONE SWITCH YOU FLIP LATER
# ---------------------------------------------------------------------------
# True  -> uses the fake "no https = phishing" rule (works right now, no model needed)
# False -> loads model.pkl from disk and uses the real trained model
#
# When your teammate gives you model.pkl, drop it in this same folder and
# change this to False. That's it — nothing else in this file needs touching
# unless your teammate's extract_features function differs from the stub
# below (see step 3).
FAKE_MODE = True

MODEL_PATH = Path(__file__).parent / "model.pkl"

# The 15 features, IN THIS EXACT ORDER, because that's the order the model
# was trained on. If you (or your teammate) ever add/remove/reorder a
# feature, the model's predictions become garbage without any error being
# raised — so treat this list as sacred.
FEATURE_ORDER = [
    "url_length", "domain_length", "num_dots", "num_hyphens",
    "num_underscore", "num_slash", "num_at_symbol", "num_digits",
    "has_ip_address", "has_https", "num_subdomains", "has_suspicious_words",
    "num_query_params", "is_shortened_url", "domain_has_digits",
]


# ---------------------------------------------------------------------------
# 3. Request/response schemas (FastAPI uses these to validate + document)
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    url: str = Field(..., example="https://example.com")


class PredictResponse(BaseModel):
    verdict: str      # "phishing" or "safe"
    confidence: float  # 0.0 - 1.0
    reasons: list[str]


class TextRequest(BaseModel):
    text: str = Field(..., example="Your account will be suspended! Click here to verify: http://bit.ly/abc")


# ---------------------------------------------------------------------------
# 4. extract_features(url) -> dict of the 15 features above
# ---------------------------------------------------------------------------
# This is a REASONABLE STUB so the app runs today. When your teammate hands
# you their real feature-extraction function, replace the body of this
# function with theirs — just make sure it still returns a dict with these
# exact 15 keys, because everything downstream (the model call, the reasons
# generator) depends on those keys existing.

SUSPICIOUS_WORDS = [
    "login", "verify", "account", "secure", "update", "banking",
    "confirm", "signin", "password", "suspend", "urgent",
]

SHORTENER_DOMAINS = [
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly",
]

IP_ADDRESS_PATTERN = re.compile(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$")


def extract_features(url: str) -> dict:
    parsed = urlparse(url if "://" in url else f"http://{url}")
    domain = parsed.netloc.split(":")[0]  # strip port if present
    path_and_query = url[len(parsed.scheme) + 3:] if parsed.scheme else url

    return {
        "url_length": len(url),
        "domain_length": len(domain),
        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "num_underscore": url.count("_"),
        "num_slash": url.count("/"),
        "num_at_symbol": url.count("@"),
        "num_digits": sum(c.isdigit() for c in url),
        "has_ip_address": int(bool(IP_ADDRESS_PATTERN.match(domain))),
        "has_https": int(parsed.scheme == "https"),
        "num_subdomains": max(domain.count(".") - 1, 0),  # example.com -> 0
        "has_suspicious_words": int(
            any(word in url.lower() for word in SUSPICIOUS_WORDS)
        ),
        "num_query_params": len(parsed.query.split("&")) if parsed.query else 0,
        "is_shortened_url": int(
            any(short in domain for short in SHORTENER_DOMAINS)
        ),
        "domain_has_digits": int(any(c.isdigit() for c in domain)),
    }


# ---------------------------------------------------------------------------
# 5a. Fake predictor (used while FAKE_MODE = True)
# ---------------------------------------------------------------------------
# Rule: no HTTPS -> phishing. This exists purely so you can build and test
# the rest of your pipeline (Chrome extension <-> this API) before the real
# model exists. It ignores the other 14 features on purpose — it's a stand-in,
# not a real detector.

def fake_predict(features: dict) -> tuple[str, float]:
    if features["has_https"] == 0:
        return "phishing", 0.91
    return "safe", 0.88


# ---------------------------------------------------------------------------
# 5b. Real predictor (used once FAKE_MODE = False)
# ---------------------------------------------------------------------------
# Loads model.pkl once at startup (not on every request — that would be slow)
# and calls model.predict()/predict_proba() on the feature vector.

_real_model = None  # cached after first load


def load_real_model():
    global _real_model
    if _real_model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail=(
                    "FAKE_MODE is False but model.pkl was not found next to "
                    "main.py. Either add model.pkl or set FAKE_MODE = True."
                ),
            )
        with open(MODEL_PATH, "rb") as f:
            _real_model = pickle.load(f)
    return _real_model


def real_predict(features: dict) -> tuple[str, float]:
    model = load_real_model()

    # Build the feature vector in the exact trained order.
    vector = [[features[name] for name in FEATURE_ORDER]]

    # model.predict() gives the class (0 = safe, 1 = phishing in most
    # phishing-detection setups — adjust the mapping below if your
    # teammate's model uses a different convention).
    prediction = model.predict(vector)[0]

    # model.predict_proba() gives the confidence, if the model supports it
    # (most scikit-learn classifiers do). Falls back to a fixed confidence
    # if the model doesn't have predict_proba.
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(vector)[0]
        confidence = float(max(probabilities))
    else:
        confidence = 0.75  # placeholder if model can't give a probability

    verdict = "phishing" if prediction == 1 else "safe"
    return verdict, confidence


# ---------------------------------------------------------------------------
# 6. Reasons generator — turns feature values into plain-English explanations
# ---------------------------------------------------------------------------
# Runs regardless of FAKE_MODE or real mode, using whatever features were
# extracted. Picks the 1-3 most suspicious-looking signals.

def generate_reasons(features: dict) -> list[str]:
    reasons = []

    if features["has_https"] == 0:
        reasons.append("no HTTPS")
    if features["has_ip_address"] == 1:
        reasons.append("uses an IP address instead of a domain name")
    if features["is_shortened_url"] == 1:
        reasons.append("uses a URL shortener")
    if features["has_suspicious_words"] == 1:
        reasons.append("contains suspicious wording (e.g. 'login', 'verify')")
    if features["num_subdomains"] >= 3:
        reasons.append("has an unusually high number of subdomains")
    if features["num_hyphens"] >= 4:
        reasons.append("has an unusually high number of hyphens in the URL")
    if features["domain_has_digits"] == 1:
        reasons.append("domain name contains digits")
    if features["url_length"] > 75:
        reasons.append("unusually long URL")
    if features["num_at_symbol"] > 0:
        reasons.append("contains an '@' symbol, which can hide the real destination")

    if not reasons:
        reasons.append("no major red flags detected in the URL structure")

    return reasons[:3]


# ---------------------------------------------------------------------------
# 7. The endpoint
# ---------------------------------------------------------------------------

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if not request.url or not request.url.strip():
        raise HTTPException(status_code=400, detail="url must not be empty")

    features = extract_features(request.url)

    if FAKE_MODE:
        verdict, confidence = fake_predict(features)
    else:
        verdict, confidence = real_predict(features)

    reasons = generate_reasons(features)

    return PredictResponse(verdict=verdict, confidence=confidence, reasons=reasons)


# ---------------------------------------------------------------------------
# 8. Text / message analyzer
# ---------------------------------------------------------------------------
# Same idea as the URL fake-predict: no trained NLP model yet, so this uses
# clear, explainable rules on things phishing messages commonly do:
# create urgency, ask for credentials/money, impersonate a brand, and
# contain a suspicious link. Every URL found inside the message is also
# run through extract_features()/generate_reasons() so link-based red flags
# show up too. Swap this function's body for a real NLP model later —
# the endpoint below doesn't need to change, same as the URL analyzer.

URGENCY_PHRASES = [
    "act now", "urgent", "immediately", "verify your account",
    "suspended", "will be closed", "limited time", "click here",
    "confirm your identity", "unusual activity", "final notice",
]

CREDENTIAL_REQUEST_PHRASES = [
    "password", "otp", "one time password", "pin number", "cvv",
    "card number", "ssn", "social security", "bank details", "login details",
]

MONEY_PHRASES = [
    "wire transfer", "gift card", "bitcoin", "claim your prize",
    "you have won", "processing fee", "refund", "lottery",
]

URL_IN_TEXT_PATTERN = re.compile(r"(https?://\S+|www\.\S+|\S+\.(?:com|net|org|in|xyz|top)\S*)", re.IGNORECASE)


def analyze_text_message(text: str) -> tuple[str, float, list[str]]:
    lowered = text.lower()
    reasons = []

    if any(phrase in lowered for phrase in URGENCY_PHRASES):
        reasons.append("uses urgency/pressure language")
    if any(phrase in lowered for phrase in CREDENTIAL_REQUEST_PHRASES):
        reasons.append("asks for sensitive credentials or personal info")
    if any(phrase in lowered for phrase in MONEY_PHRASES):
        reasons.append("mentions money, prizes, or payment in a suspicious way")

    # If the message contains a link, run it through the same URL analyzer
    # logic so link-based red flags (no https, IP address, shortener, etc.)
    # are included too.
    found_urls = URL_IN_TEXT_PATTERN.findall(text)
    if found_urls:
        first_url = found_urls[0]
        url_features = extract_features(first_url)
        url_reasons = generate_reasons(url_features)
        if url_reasons and url_reasons[0] != "no major red flags detected in the URL structure":
            reasons.append(f"contains a suspicious link ({url_reasons[0]})")

    if reasons:
        verdict = "phishing"
        confidence = min(0.99, 0.6 + 0.1 * len(reasons))
    else:
        verdict = "safe"
        confidence = 0.85

    if not reasons:
        reasons.append("no major red flags detected in the message wording")

    return verdict, confidence, reasons[:3]


@app.post("/analyze-text", response_model=PredictResponse)
def analyze_text(request: TextRequest):
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")

    verdict, confidence, reasons = analyze_text_message(request.text)
    return PredictResponse(verdict=verdict, confidence=confidence, reasons=reasons)


# ---------------------------------------------------------------------------
# 9. Image analyzer
# ---------------------------------------------------------------------------
# STUB / PLACEHOLDER, clearly labeled as such. There is no trained image
# model, and real phishing-screenshot detection normally needs one (or an
# OCR step to pull text out of the image and re-run it through the text
# analyzer above). To keep this working today with zero extra setup, this
# stub only looks at basic image metadata (file size, format) as a
# stand-in signal, and always says so in the reasons. Replace the body of
# analyze_image_stub() once you have a real image model or add OCR
# (e.g. pytesseract) to extract and analyze on-image text for real.

def analyze_image_stub(filename: str, content_length: int) -> tuple[str, float, list[str]]:
    reasons = ["image analysis is a placeholder — no trained model wired in yet"]

    # Very rough, non-predictive heuristic just so the endpoint returns
    # something meaningful today: unusually small "screenshot" files are
    # flagged as worth a second look. This is NOT a real detection signal.
    if content_length < 15_000:
        reasons.append("unusually small image file size")
        return "phishing", 0.55, reasons

    return "safe", 0.55, reasons


@app.post("/analyze-image", response_model=PredictResponse)
async def analyze_image(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="uploaded file must be an image")

    contents = await file.read()
    verdict, confidence, reasons = analyze_image_stub(file.filename, len(contents))
    return PredictResponse(verdict=verdict, confidence=confidence, reasons=reasons)


@app.get("/")
def health_check():
    return {"status": "ok", "fake_mode": FAKE_MODE}
