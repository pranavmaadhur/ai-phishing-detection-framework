"""
train_model.py
---------------
Trains an XGBoost classifier to detect phishing URLs from 15 pre-extracted
numeric features, evaluates it on a held-out test set, and saves the
trained model to model.pkl.

Usage:
    python3 train_model.py

Expects train.csv and test.csv in the same folder, each with these columns:
    url_length, domain_length, num_dots, num_hyphens, num_underscore,
    num_slash, num_at_symbol, num_digits, has_ip_address, has_https,
    num_subdomains, has_suspicious_words, num_query_params,
    is_shortened_url, domain_has_digits, label
"""

import pickle
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

FEATURE_COLUMNS = [
    "url_length", "domain_length", "num_dots", "num_hyphens",
    "num_underscore", "num_slash", "num_at_symbol", "num_digits",
    "has_ip_address", "has_https", "num_subdomains",
    "has_suspicious_words", "num_query_params", "is_shortened_url",
    "domain_has_digits",
]
LABEL_COLUMN = "label"


def load_data(train_path="train.csv", test_path="test.csv"):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    missing_train = set(FEATURE_COLUMNS + [LABEL_COLUMN]) - set(train_df.columns)
    missing_test = set(FEATURE_COLUMNS + [LABEL_COLUMN]) - set(test_df.columns)
    if missing_train:
        raise ValueError(f"train.csv is missing columns: {missing_train}")
    if missing_test:
        raise ValueError(f"test.csv is missing columns: {missing_test}")

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[LABEL_COLUMN]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[LABEL_COLUMN]
    return X_train, y_train, X_test, y_test


def train_model(X_train, y_train):
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("=" * 55)
    print("MODEL PERFORMANCE ON TEST SET")
    print("=" * 55)
    print(f"Accuracy:  {acc:.4f}  ({acc*100:.2f}%)")
    print(f"Precision: {prec:.4f}  ({prec*100:.2f}%)")
    print(f"Recall:    {rec:.4f}  ({rec*100:.2f}%)")
    print(f"F1 score:  {f1:.4f}")
    print("\nConfusion matrix:")
    print("                Predicted Legit   Predicted Phishing")
    print(f"Actual Legit         {cm[0][0]:>6}              {cm[0][1]:>6}")
    print(f"Actual Phishing      {cm[1][0]:>6}              {cm[1][1]:>6}")
    print("\nFull classification report:")
    print(classification_report(y_test, y_pred, target_names=["legit", "phishing"]))

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "confusion_matrix": cm}


def show_feature_importance(model):
    importances = model.feature_importances_
    ranked = sorted(zip(FEATURE_COLUMNS, importances), key=lambda x: x[1], reverse=True)

    print("=" * 55)
    print("FEATURE IMPORTANCE (which signals matter most)")
    print("=" * 55)
    for name, score in ranked:
        bar = "#" * int(score * 100)
        print(f"{name:<22} {score:.4f}  {bar}")

    return ranked


def save_model(model, path="model.pkl"):
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"\nModel saved to {path}")


if __name__ == "__main__":
    print("Loading data...")
    X_train, y_train, X_test, y_test = load_data()
    print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}")

    print("\nTraining XGBoost classifier...")
    model = train_model(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test)
    ranked_features = show_feature_importance(model)

    save_model(model, "model.pkl")
