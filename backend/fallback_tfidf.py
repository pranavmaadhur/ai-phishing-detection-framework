"""
FALLBACK for Part A - use only if the DistilBERT download in main.py is
too slow/unreliable on venue wifi.

This is TF-IDF (turns text into word/phrase-frequency numbers) feeding a
small neural network (scikit-learn's MLPClassifier - a couple of dense
layers, which is the same building block deep learning uses, just via
sklearn instead of PyTorch/Keras). No download required beyond `pip
install scikit-learn`, and it trains in about 1 second on the small
sample dataset below.

This ships with a small hand-written example dataset so it works
out-of-the-box for a demo. For a stronger model with more time:
    pip install datasets
    from datasets import load_dataset
    ds = load_dataset("ealvaradob/phishing-dataset", "combined_reduced")
This is a public Hugging Face dataset combining phishing emails, SMS,
and website text (already cleaned/deduplicated) - swap it in for
PHISHING_EXAMPLES / SAFE_EXAMPLES below and retrain.

Same function signature as main.py's analyze_text(), so you can import
whichever version you need without touching the FastAPI route.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier

# A small, hand-written example set covering common phishing phrasing.
# Swap this for a real dataset (see docstring above) if you have 10+
# extra minutes before the demo - more examples = better accuracy.
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

# Word + 2-word-phrase counts, weighted by how distinctive each is.
_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
_X = _vectorizer.fit_transform(_texts)

# One small hidden layer - enough to combine TF-IDF features non-linearly
# without overfitting on a tiny dataset.
_clf = MLPClassifier(hidden_layer_sizes=(16,), max_iter=2000, random_state=42)
_clf.fit(_X, _labels)


def analyze_text(text: str) -> dict:
    """
    Reads page/message TEXT and flags phishing language.
    Returns: {"verdict": "phishing" | "safe", "confidence": 0.0-1.0}
    """
    if not text or not text.strip():
        return {"verdict": "safe", "confidence": 0.0}

    X = _vectorizer.transform([text])
    safe_prob, phishing_prob = _clf.predict_proba(X)[0]

    if phishing_prob >= safe_prob:
        return {"verdict": "phishing", "confidence": round(float(phishing_prob), 4)}
    else:
        return {"verdict": "safe", "confidence": round(float(safe_prob), 4)}


if __name__ == "__main__":
    # quick smoke test
    print(analyze_text("Your account has been suspended, click immediately to verify."))
    print(analyze_text("Let's grab coffee sometime this week."))
