"""
Train Constraint-Aware Router Policy for MixRAGRec
--------------------------------------------------
Trains a calibrated classifier on TRAIN set (303 queries),
tunes the decision threshold on DEV set (99 queries) based on the
utility-cost trade-off (analogous to MMAPO reward in the paper),
and saves the trained model to data/router_model.joblib.
"""

import json
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score

FEATURE_NAMES = [
    "frac_title_gb",
    "max_price_k",
    "min_ram_16",
    "min_storage_512",
    "has_storage",
    "has_brand",
    "n_hard",
    "soft_light",
    "soft_large_screen",
    "soft_small_screen",
]


def extract_features(row):
    f = row["feats"]
    return [
        f["frac_title_gb"],
        (f["max_price"] or 1000.0) / 1000.0,
        (f["min_ram_gb"] or 8.0) / 16.0,
        (f["min_storage_gb"] or 0.0) / 512.0,
        1.0 if f["min_storage_gb"] is not None else 0.0,
        1.0 if f["has_brand"] else 0.0,
        float(f["n_hard"]),
        1.0 if f["soft"] == "light" else 0.0,
        1.0 if f["soft"] == "large_screen" else 0.0,
        1.0 if f["soft"] == "small_screen" else 0.0,
    ]


def eval_policy(rows, probs, threshold):
    n = len(rows)
    correct = 0
    tokens = 0
    e1_chosen = 0
    for r, p in zip(rows, probs):
        choice = "E1" if p >= threshold else "E2"
        if choice == "E1":
            e1_chosen += 1
        correct += 1 if r["out"][choice]["ok"] else 0
        tokens += r["out"][choice]["toks"]
    return correct / n, tokens / n, e1_chosen


def main():
    print("=" * 60)
    print("Training Constraint-Aware Router (MixRAGRec)")
    print("=" * 60)

    data = [
        json.loads(line)
        for line in open("data/router_data.jsonl", "r", encoding="utf-8")
    ]

    train_rows = [r for r in data if r["split"] == "train"]
    dev_rows = [r for r in data if r["split"] == "dev"]
    test_rows = [r for r in data if r["split"] == "test"]

    print(
        f"Data splits: Train={len(train_rows)}, Dev={len(dev_rows)}, Test={len(test_rows)}"
    )

    X_train = np.array([extract_features(r) for r in train_rows])
    y_train = np.array([1 if r["out"]["E1"]["ok"] else 0 for r in train_rows])

    X_dev = np.array([extract_features(r) for r in dev_rows])
    y_dev = np.array([1 if r["out"]["E1"]["ok"] else 0 for r in dev_rows])

    X_test = np.array([extract_features(r) for r in test_rows])
    y_test = np.array([1 if r["out"]["E1"]["ok"] else 0 for r in test_rows])

    # Fit scaler on Train only (no data leakage)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_dev_s = scaler.transform(X_dev)
    X_test_s = scaler.transform(X_test)

    # Train Calibrated Logistic Regression
    clf = LogisticRegression(C=1.0, random_state=42, max_iter=1000)
    clf.fit(X_train_s, y_train)

    p_train = clf.predict_proba(X_train_s)[:, 1]
    p_dev = clf.predict_proba(X_dev_s)[:, 1]
    p_test = clf.predict_proba(X_test_s)[:, 1]

    print("\n--- Model Performance (ROC-AUC) ---")
    print(f"Train AUC: {roc_auc_score(y_train, p_train):.3f}")
    print(f"Dev AUC:   {roc_auc_score(y_dev, p_dev):.3f}")
    print(f"Test AUC:  {roc_auc_score(y_test, p_test):.3f}")

    print("\n--- Learned Feature Weights (Interpretability) ---")
    weights = sorted(
        zip(FEATURE_NAMES, clf.coef_[0]), key=lambda x: abs(x[1]), reverse=True
    )
    for name, w in weights:
        direction = "Favors E1 (cheap)" if w > 0 else "Favors E2 (detailed)"
        print(f"  {name:20s}: {w:+.3f}  --> {direction}")

    # Threshold Optimization on Dev
    print("\n--- Threshold Sweep on Dev (Selecting Optimal Trade-off) ---")
    print(f"{'Threshold':10s} | {'Dev Acc':10s} | {'Avg Tokens':12s} | {'E1 Rate':10s}")
    print("-" * 52)

    best_thresh = 0.88
    best_utility = -1e9
    lambda_cost = (
        0.15  # Cost weight balancing accuracy vs normalized token overhead
    )

    for t in np.linspace(0.60, 0.98, 20):
        acc, toks, e1_count = eval_policy(dev_rows, p_dev, t)
        # Utility matching MMAPO objective: Accuracy - lambda * (tokens / max_tokens)
        utility = acc - lambda_cost * (toks / 2442.0)
        if utility > best_utility:
            best_utility = utility
            best_thresh = t
        print(
            f"{t:10.2f} | {acc:9.1%}  | {toks:10.0f}   | {e1_count/len(dev_rows):9.1%}"
        )

    print(f"\nSelected optimal threshold on Dev: {best_thresh:.2f}")

    # Save model and artifacts
    model_bundle = {
        "classifier": clf,
        "scaler": scaler,
        "feature_names": FEATURE_NAMES,
        "threshold": best_thresh,
    }
    joblib.dump(model_bundle, "data/router_model.joblib")
    print(f"Saved router model bundle -> data/router_model.joblib")


if __name__ == "__main__":
    main()
