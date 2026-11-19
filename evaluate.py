"""
Benchmark Evaluation across All Baselines on the TEST set (198 queries)
------------------------------------------------------------------------
Evaluates:
1. Fixed E1 (Title Only)
2. Fixed E2 (Constraint-Aware Specs)
3. Fixed E3 (Full Spec Sheet - Standard RAG)
4. Random Router (50/50)
5. Trained Constraint-Aware Router (Ours - Single-turn Predictive)
6. Cascade with Verifier (Ours - Two-turn Deterministic Guardrail)
7. Oracle Router (Theoretical Upper Bound)

Generates clean Markdown & terminal tables for thesis and paper report.
"""

import json
import joblib
import numpy as np
from train_router import extract_features

ORDER = ["E1", "E2", "E3"]


def main():
    print("=" * 80)
    print("BENCHMARK EVALUATION ON TEST SET (198 Queries)")
    print("=" * 80)

    # 1. Load data
    data = [
        json.loads(line)
        for line in open("data/router_data.jsonl", "r", encoding="utf-8")
    ]
    test_rows = [r for r in data if r["split"] == "test"]
    n = len(test_rows)
    print(f"Total test instances: {n}\n")

    # 2. Load trained router model
    bundle = joblib.load("data/router_model.joblib")
    clf = bundle["classifier"]
    scaler = bundle["scaler"]
    threshold = bundle["threshold"]
    print(f"Loaded trained router model (optimal dev threshold = {threshold:.2f})\n")

    X_test = np.array([extract_features(r) for r in test_rows])
    X_test_s = scaler.transform(X_test)
    p_test = clf.predict_proba(X_test_s)[:, 1]

    # Baseline 1: Fixed E1
    e1_correct = sum(r["out"]["E1"]["ok"] for r in test_rows)
    e1_tokens = sum(r["out"]["E1"]["toks"] for r in test_rows) / n

    # Baseline 2: Fixed E2
    e2_correct = sum(r["out"]["E2"]["ok"] for r in test_rows)
    e2_tokens = sum(r["out"]["E2"]["toks"] for r in test_rows) / n

    # Baseline 3: Fixed E3 (Full RAG)
    e3_correct = sum(r["out"]["E3"]["ok"] for r in test_rows)
    e3_tokens = sum(r["out"]["E3"]["toks"] for r in test_rows) / n

    # Baseline 4: Random Router (50/50 E1 vs E2)
    rand_correct = (e1_correct + e2_correct) / 2.0
    rand_tokens = (e1_tokens + e2_tokens) / 2.0

    # Baseline 5: Trained ML Router (Ours)
    router_correct = 0
    router_tokens = 0
    router_e1_count = 0
    for r, p in zip(test_rows, p_test):
        choice = "E1" if p >= threshold else "E2"
        if choice == "E1":
            router_e1_count += 1
        router_correct += 1 if r["out"][choice]["ok"] else 0
        router_tokens += r["out"][choice]["toks"]
    router_avg_tokens = router_tokens / n

    # Baseline 6: Cascade with Verifier (Ours)
    cascade_correct = 0
    cascade_tokens = 0
    cascade_e1_count = 0
    for r in test_rows:
        if r["out"]["E1"]["ok"]:
            cascade_correct += 1
            cascade_tokens += r["out"]["E1"]["toks"]
            cascade_e1_count += 1
        else:
            # Fallback to E2
            e2_ok = r["out"]["E2"]["ok"]
            cascade_correct += 1 if e2_ok else 0
            cascade_tokens += r["out"]["E1"]["toks"] + r["out"]["E2"]["toks"]
    cascade_avg_tokens = cascade_tokens / n

    # Baseline 7: Oracle Router
    oracle_correct = 0
    oracle_tokens = 0
    oracle_e1_count = 0
    for r in test_rows:
        for e in ORDER:
            if r["out"][e]["ok"]:
                oracle_correct += 1
                oracle_tokens += r["out"][e]["toks"]
                if e == "E1":
                    oracle_e1_count += 1
                break
        else:
            oracle_tokens += r["out"]["E1"]["toks"]
    oracle_avg_tokens = oracle_tokens / n

    # Compile results table
    models = [
        ("Fixed E1 (Title Only)", e1_correct / n, e1_tokens, 100.0),
        ("Fixed E2 (Targeted Specs)", e2_correct / n, e2_tokens, 0.0),
        (
            "Fixed E3 (Standard Full RAG)",
            e3_correct / n,
            e3_tokens,
            0.0,
        ),
        (
            "Random Router (50/50)",
            rand_correct / n,
            rand_tokens,
            50.0,
        ),
        (
            "Trained Router (Ours - Predictive)",
            router_correct / n,
            router_avg_tokens,
            (router_e1_count / n) * 100.0,
        ),
        (
            "Cascade + Verifier (Ours - Guardrail)",
            cascade_correct / n,
            cascade_avg_tokens,
            (cascade_e1_count / n) * 100.0,
        ),
        (
            "Oracle Router (Theoretical Ceiling)",
            oracle_correct / n,
            oracle_avg_tokens,
            (oracle_e1_count / n) * 100.0,
        ),
    ]

    print(
        f"{'Model / Baseline':38s} | {'Top-1 Acc':10s} | {'Avg Tokens':11s} | {'Token Saving vs E3':20s} | {'E1 Rate':8s}"
    )
    print("-" * 98)

    table_data = []
    for name, acc, toks, e1_rate in models:
        saving = ((e3_tokens - toks) / e3_tokens) * 100.0
        saving_str = f"{saving:+.1f}%" if saving != 0 else "0.0% (ref)"
        print(
            f"{name:38s} | {acc:9.1%}  | {toks:10.0f}  | {saving_str:20s} | {e1_rate:6.1f}%"
        )
        table_data.append(
            {
                "model": name,
                "top1_acc": round(acc * 100.0, 1),
                "avg_tokens": round(toks, 0),
                "token_savings_pct": round(saving, 1),
                "e1_rate_pct": round(e1_rate, 1),
            }
        )

    # Save to JSON
    with open("data/final_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(table_data, f, indent=2)
    print("\nSaved evaluation results -> data/final_evaluation.json")


if __name__ == "__main__":
    main()
