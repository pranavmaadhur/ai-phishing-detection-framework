"""
Merged Phishing Detection API
=============================
POST /predict       -> URL analyzer + OpenPhish threat intelligence
POST /predict_text  -> TF-IDF/MLP text analyzer
POST /analyze-text  -> existing rule-based text analyzer
POST /analyze-image -> image analyzer placeholder
"""

import re
import pickle
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier


# ---------------------------------------------------------------------------
# 1. App setup + CORS
# ---------------------------------------------------------------------------

app = FastAPI(title="Phishing Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# 2. URL model settings
# ---------------------------------------------------------------------------

FAKE_MODE = True
MODEL_PATH = Path(__file__).parent / "model.pkl"

FEATURE_ORDER = [
    "url_length", "domain_length", "num_dots", "num_hyphens",
    "num_underscore", "num_slash", "num_at_symbol", "num_digits",
    "has_ip_address", "has_https", "num_subdomains", "has_suspicious_words",
    "num_query_params", "is_shortened_url", "domain_has_digits",
]


# ---------------------------------------------------------------------------
# 3. Request/response schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    url: str = Field(..., example="https://example.com")


class PredictResponse(BaseModel):
    verdict: str
    confidence: float
    reasons: list[str]
    known_phishing: bool = False
    threat_intel_source: str = "OpenPhish"


class TextRequest(BaseModel):
    text: str = Field(
        ...,
        example="Your account will be suspended! Click here to verify."
    )


# ---------------------------------------------------------------------------
# 4. URL feature extraction
# ---------------------------------------------------------------------------

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
    domain = parsed.netloc.split(":")[0]

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
        "num_subdomains": max(domain.count(".") - 1, 0),
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
# 5. URL predictors
# ---------------------------------------------------------------------------

def fake_predict(features: dict) -> tuple[str, float]:
    if features["has_https"] == 0:
        return "phishing", 0.91
    return "safe", 0.88


_real_model = None


def load_real_model():
    global _real_model

    if _real_model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail=(
                    "FAKE_MODE is False but model.pkl was not found next to "
                    "this Python file. Either add model.pkl or set FAKE_MODE = True."
                ),
            )

        with open(MODEL_PATH, "rb") as f:
            _real_model = pickle.load(f)

    return _real_model


def real_predict(features: dict) -> tuple[str, float]:
    model = load_real_model()
    vector = [[features[name] for name in FEATURE_ORDER]]

    prediction = model.predict(vector)[0]

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(vector)[0]
        confidence = float(max(probabilities))
    else:
        confidence = 0.75

    verdict = "phishing" if prediction == 1 else "safe"
    return verdict, confidence


# ---------------------------------------------------------------------------
# 6. URL reasons
# ---------------------------------------------------------------------------

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
# 7. OpenPhish threat intelligence
# ---------------------------------------------------------------------------

OPENPHISH_FEED_URL = "https://openphish.com/feed.txt"
CACHE_TTL_SECONDS = 3 * 60 * 60
FETCH_TIMEOUT_SECONDS = 5

_threat_cache = {
    "urls": set(),
    "domains": set(),
    "fetched_at": 0.0,
}


def _domain_of(url: str) -> str:
    try:
        netloc = urlparse(
            url if "://" in url else f"http://{url}"
        ).netloc
        return netloc.lower().split(":")[0]
    except Exception:
        return ""


def _fetch_threat_feed() -> None:
    resp = requests.get(
        OPENPHISH_FEED_URL,
        timeout=FETCH_TIMEOUT_SECONDS
    )
    resp.raise_for_status()

    urls = set()
    domains = set()

    for line in resp.text.splitlines():
        line = line.strip()
        if not line:
            continue

        urls.add(line)

        domain = _domain_of(line)
        if domain:
            domains.add(domain)

    _threat_cache["urls"] = urls
    _threat_cache["domains"] = domains
    _threat_cache["fetched_at"] = time.time()


def _ensure_fresh_threat_cache() -> bool:
    is_stale = (
        time.time() - _threat_cache["fetched_at"]
    ) > CACHE_TTL_SECONDS

    if not is_stale and _threat_cache["fetched_at"] > 0:
        return True

    try:
        _fetch_threat_feed()
        return True
    except Exception:
        # If an old cache exists, keep using it.
        return _threat_cache["fetched_at"] > 0


def check_threat_intel(url: str) -> dict:
    """
    Checks whether the URL or its domain appears in the OpenPhish feed.
    """
    if not url or not url.strip():
        return {
            "known_phishing": False,
            "source": "OpenPhish",
        }

    cache_ok = _ensure_fresh_threat_cache()

    if not cache_ok:
        return {
            "known_phishing": False,
            "source": "OpenPhish (feed unavailable)",
        }

    url = url.strip()

    if url in _threat_cache["urls"]:
        return {
            "known_phishing": True,
            "source": "OpenPhish",
        }

    domain = _domain_of(url)

    if domain and domain in _threat_cache["domains"]:
        return {
            "known_phishing": True,
            "source": "OpenPhish",
        }

    return {
        "known_phishing": False,
        "source": "OpenPhish",
    }


# ---------------------------------------------------------------------------
# 8. URL endpoint
# ---------------------------------------------------------------------------

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if not request.url or not request.url.strip():
        raise HTTPException(status_code=400, detail="url must not be empty")

    url = request.url.strip()
    features = extract_features(url)

    if FAKE_MODE:
        verdict, confidence = fake_predict(features)
    else:
        verdict, confidence = real_predict(features)

    reasons = generate_reasons(features)

    # Extra signal from OpenPhish.
    threat_intel = check_threat_intel(url)

    if threat_intel["known_phishing"]:
        verdict = "phishing"
        confidence = max(confidence, 0.99)
        reasons.insert(0, "URL/domain is listed in the OpenPhish phishing feed")
        reasons = reasons[:3]

    return PredictResponse(
        verdict=verdict,
        confidence=confidence,
        reasons=reasons,
        known_phishing=threat_intel["known_phishing"],
        threat_intel_source=threat_intel["source"],
    )


# ---------------------------------------------------------------------------
# 9. TF-IDF + MLP text analyzer
# ---------------------------------------------------------------------------

PHISHING_EXAMPLES = [
    "Your account has been suspended. Verify your identity immediately by clicking the link below.",
    "URGENT: Unusual sign-in activity detected. Confirm your password now to avoid permanent lock.",
    "Dear customer, your payment failed. Update your billing information within 24 hours or lose access.",
    "Security Alert: Your account will be closed. Click here to verify your account now.",
    "Congratulations! You have won a prize. Claim it now by entering your bank details.",
    "Action required: Your package could not be delivered. Confirm your address to reschedule immediately.",
    "We detected suspicious activity on your account. Log in now to avoid suspension.",
    "Your subscription has expired. Renew now to avoid losing access to your files.",
    "This is your final notice. Verify your account today or it will be permanently deleted.",
    "Click here immediately to claim your refund before it expires.",
]

SAFE_EXAMPLES = [
    "Hey, are we still on for lunch tomorrow at noon?",
    "The quarterly report is attached, let me know if you have questions.",
    "Thanks for your order! Your package will arrive in 3-5 business days.",
    "Reminder: team meeting moved to 3pm in the main conference room.",
    "Here's the recipe you asked for, hope you enjoy it.",
    "Your monthly statement is now available to view in your account.",
    "Great catching up with you last week, let's do it again soon.",
    "The document you requested is linked below, no rush on reviewing it.",
    "Happy birthday! Hope you have a wonderful day.",
    "Here's the agenda for tomorrow's standup.",
]

_texts = PHISHING_EXAMPLES + SAFE_EXAMPLES
_labels = [1] * len(PHISHING_EXAMPLES) + [0] * len(SAFE_EXAMPLES)

_text_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
_text_X = _text_vectorizer.fit_transform(_texts)

_text_clf = MLPClassifier(
    hidden_layer_sizes=(16,),
    max_iter=2000,
    random_state=42,
)
_text_clf.fit(_text_X, _labels)


def analyze_text_ml(text: str) -> dict:
    if not text or not text.strip():
        return {
            "verdict": "safe",
            "confidence": 0.0,
        }

    X = _text_vectorizer.transform([text])
    safe_prob, phishing_prob = _text_clf.predict_proba(X)[0]

    if phishing_prob >= safe_prob:
        return {
            "verdict": "phishing",
            "confidence": round(float(phishing_prob), 4),
        }

    return {
        "verdict": "safe",
        "confidence": round(float(safe_prob), 4),
    }


@app.post("/predict_text", response_model=PredictResponse)
def predict_text(request: TextRequest):
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")

    result = analyze_text_ml(request.text)

    return PredictResponse(
        verdict=result["verdict"],
        confidence=result["confidence"],
        reasons=["TF-IDF + MLP text analysis"],
        known_phishing=False,
        threat_intel_source="Not applicable",
    )


# ---------------------------------------------------------------------------
# 10. Existing rule-based text analyzer
# ---------------------------------------------------------------------------

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

URL_IN_TEXT_PATTERN = re.compile(
    r"(https?://\S+|www\.\S+|\S+\.(?:com|net|org|in|xyz|top)\S*)",
    re.IGNORECASE,
)


def analyze_text_message(text: str) -> tuple[str, float, list[str]]:
    lowered = text.lower()
    reasons = []

    if any(phrase in lowered for phrase in URGENCY_PHRASES):
        reasons.append("uses urgency/pressure language")

    if any(phrase in lowered for phrase in CREDENTIAL_REQUEST_PHRASES):
        reasons.append("asks for sensitive credentials or personal info")

    if any(phrase in lowered for phrase in MONEY_PHRASES):
        reasons.append("mentions money, prizes, or payment in a suspicious way")

    found_urls = URL_IN_TEXT_PATTERN.findall(text)

    if found_urls:
        first_url = found_urls[0]
        url_features = extract_features(first_url)
        url_reasons = generate_reasons(url_features)

        if (
            url_reasons
            and url_reasons[0]
            != "no major red flags detected in the URL structure"
        ):
            reasons.append(
                f"contains a suspicious link ({url_reasons[0]})"
            )

    if reasons:
        verdict = "phishing"
        confidence = min(0.99, 0.6 + 0.1 * len(reasons))
    else:
        verdict = "safe"
        confidence = 0.85
        reasons.append(
            "no major red flags detected in the message wording"
        )

    return verdict, confidence, reasons[:3]


@app.post("/analyze-text", response_model=PredictResponse)
def analyze_text(request: TextRequest):
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")

    verdict, confidence, reasons = analyze_text_message(request.text)

    return PredictResponse(
        verdict=verdict,
        confidence=confidence,
        reasons=reasons,
        known_phishing=False,
        threat_intel_source="Not checked",
    )


# ---------------------------------------------------------------------------
# 11. Image analyzer placeholder
# ---------------------------------------------------------------------------

def analyze_image_stub(
    filename: str,
    content_length: int
) -> tuple[str, float, list[str]]:
    reasons = [
        "image analysis is a placeholder — no trained model wired in yet"
    ]

    if content_length < 15_000:
        reasons.append("unusually small image file size")
        return "phishing", 0.55, reasons

    return "safe", 0.55, reasons


@app.post("/analyze-image", response_model=PredictResponse)
async def analyze_image(file: UploadFile = File(...)):
    if (
        not file.content_type
        or not file.content_type.startswith("image/")
    ):
        raise HTTPException(
            status_code=400,
            detail="uploaded file must be an image",
        )

    contents = await file.read()
    verdict, confidence, reasons = analyze_image_stub(
        file.filename,
        len(contents),
    )

    return PredictResponse(
        verdict=verdict,
        confidence=confidence,
        reasons=reasons,
        known_phishing=False,
        threat_intel_source="Not applicable",
    )


# ---------------------------------------------------------------------------
# 12. Health check
# ---------------------------------------------------------------------------

@app.get("/")
def health_check():
    return {
        "status": "ok",
        "fake_mode": FAKE_MODE,
    }
