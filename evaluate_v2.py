"""
Evaluation v2 for Constraint-Aware MixRAGRec
--------------------------------------------
Adds over evaluate.py:
  1. Deterministic filter baseline (no LLM ranking), with GT and extracted constraints
  2. Wilson 95% CIs for accuracy + paired bootstrap for method differences
  3. Extractor token cost + LLM-call count (latency proxy)
  4. Extractor-error audit (cases where the Verifier can be fooled)

Run:  python evaluate_v2.py
"""

import json
import joblib
import numpy as np

from train_router import extract_features
from experts import cards
from extractor import extract

ORDER = ["E1", "E2", "E3"]

# Approx. prompt tokens of one extractor call (SYSTEM prompt + request).
# Replace with a measured value (usage.prompt_tokens) if you log it.
EXT_TOKENS = 450

RNG = np.random.default_rng(42)
N_BOOT = 10_000


# ---------------------------------------------------------------- stats
def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return center - half, center + half


def paired_bootstrap(a, b, n_boot=N_BOOT):
    """CI of mean(a) - mean(b) over per-query 0/1 correctness vectors."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = len(a)
    idx = RNG.integers(0, n, size=(n_boot, n))
    diffs = a[idx].mean(1) - b[idx].mean(1)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(a.mean() - b.mean()), float(lo), float(hi)


# ---------------------------------------------------- constraint helpers
def satisfies(item, c):
    """c: dict with max_price/min_ram_gb/min_storage_gb/brand (None = unused)."""
    if c.get("max_price") is not None and (
        item["price"] is None or item["price"] > c["max_price"]
    ):
        return False
    if c.get("min_ram_gb") is not None and (
        item["ram_gb"] is None or item["ram_gb"] < c["min_ram_gb"]
    ):
        return False
    if c.get("min_storage_gb") is not None and (
        item["storage_gb"] is None or item["storage_gb"] < c["min_storage_gb"]
    ):
        return False
    if c.get("brand") is not None and (item["brand"] or "").lower() != str(
        c["brand"]
    ).lower():
        return False
    return True


def deterministic_pick(pool, constraints, soft):
    """Filter pool by hard constraints, order survivors by soft preference."""
    ok = [i for i in pool if satisfies(cards[i], constraints)]
    if not ok:
        return None
    if soft == "light":
        ok.sort(key=lambda i: cards[i]["weight_kg"] if cards[i]["weight_kg"] is not None else 99)
    elif soft == "large_screen":
        ok.sort(key=lambda i: -(cards[i]["screen_in"] or 0))
    elif soft == "small_screen":
        ok.sort(key=lambda i: cards[i]["screen_in"] if cards[i]["screen_in"] is not None else 99)
    return ok[0]


def gt_constraints(q):
    c = q["constraints"]
    return {
        "max_price": c.get("max_price"),
        "min_ram_gb": c.get("min_ram_gb"),
        "min_storage_gb": c.get("min_storage_gb"),
        "brand": c.get("brand"),
    }


# ----------------------------------------------------------------- main
def main():
    rows = [json.loads(l) for l in open("data/router_data.jsonl", encoding="utf-8")]
    test_rows = [r for r in rows if r["split"] == "test"]
    qs = {
        q["qid"]: q
        for q in (json.loads(l) for l in open("data/queries.jsonl", encoding="utf-8"))
    }
    n = len(test_rows)

    bundle = joblib.load("data/router_model.joblib")
    clf, scaler, thr = bundle["classifier"], bundle["scaler"], bundle["threshold"]
    p = clf.predict_proba(scaler.transform(np.array([extract_features(r) for r in test_rows])))[:, 1]

    # per-method: (correct vector, prompt tokens vector, extractor calls, llm ranking calls)
    M = {}

    def add(name, correct, toks, ext_calls, rank_calls):
        M[name] = {
            "correct": np.array(correct, float),
            "toks": np.array(toks, float),
            "ext_calls": np.array(ext_calls, float),
            "rank_calls": np.array(rank_calls, float),
        }

    for e in ORDER:
        add(
            f"Fixed {e}",
            [r["out"][e]["ok"] for r in test_rows],
            [r["out"][e]["toks"] for r in test_rows],
            [0 if e != "E2" else 1] * n,
            [1] * n,
        )

    # Trained router (predictive): extractor + 1 ranking call
    c, t = [], []
    for r, pr in zip(test_rows, p):
        ch = "E1" if pr >= thr else "E2"
        c.append(r["out"][ch]["ok"])
        t.append(r["out"][ch]["toks"])
    add("Trained Router", c, t, [1] * n, [1] * n)

    # Cascade: verifier decision = GT-validity of E1's top-1 (see audit below)
    c, t, calls = [], [], []
    for r in test_rows:
        if r["out"]["E1"]["ok"]:
            c.append(1)
            t.append(r["out"]["E1"]["toks"])
            calls.append(1)
        else:
            c.append(r["out"]["E2"]["ok"])
            t.append(r["out"]["E1"]["toks"] + r["out"]["E2"]["toks"])
            calls.append(2)
    add("Cascade + Verifier", c, t, [1] * n, calls)

    # Oracle
    c, t = [], []
    for r in test_rows:
        for e in ORDER:
            if r["out"][e]["ok"]:
                c.append(1)
                t.append(r["out"][e]["toks"])
                break
        else:
            c.append(0)
            t.append(r["out"]["E1"]["toks"])
    add("Oracle Router", c, t, [0] * n, [1] * n)

    # ---- Deterministic filter baselines (no LLM ranking at all) ----
    c_gt, c_ex = [], []
    ext_mismatch = []
    for r in test_rows:
        q = qs[r["qid"]]
        valid = set(q["valid_in_pool"])
        gt = gt_constraints(q)

        pick = deterministic_pick(q["pool"], gt, q["soft_pref"])
        c_gt.append(int(pick in valid) if pick is not None else 0)

        ext, _ = extract(q["query"])
        if ext is None:
            c_ex.append(0)
            ext_mismatch.append(r["qid"])
            continue
        h = ext.hard
        ex = {
            "max_price": h.max_price,
            "min_ram_gb": h.min_ram_gb,
            "min_storage_gb": h.min_storage_gb,
            "brand": h.brand,
        }
        pick = deterministic_pick(q["pool"], ex, ext.soft_pref)
        c_ex.append(int(pick in valid) if pick is not None else 0)

        same = all(
            (gt[k] is None and ex[k] is None)
            or (
                gt[k] is not None
                and ex[k] is not None
                and str(gt[k]).lower() == str(ex[k]).lower()
                if k == "brand"
                else gt[k] == ex[k]
            )
            for k in gt
        )
        if not same:
            ext_mismatch.append(r["qid"])

    add("Filter only (GT constraints, upper bound)", c_gt, [0] * n, [0] * n, [0] * n)
    add("Filter only (extracted constraints)", c_ex, [0] * n, [1] * n, [0] * n)

    # ---------------------------------------------------------- report
    e3_adj = M["Fixed E3"]["toks"].mean()
    print("=" * 118)
    print(f"EVALUATION v2 — TEST SET (n={n}), extractor cost = {EXT_TOKENS} tok/call")
    print("=" * 118)
    print(
        f"{'Method':44s} | {'Acc':6s} | {'95% CI (Wilson)':16s} | {'Tok(LLM rank)':13s} | "
        f"{'Tok(+extractor)':15s} | {'Saving':7s} | {'LLM calls':9s}"
    )
    print("-" * 118)

    out = []
    for name, m in M.items():
        k = int(m["correct"].sum())
        lo, hi = wilson(k, n)
        tok_rank = m["toks"].mean()
        tok_adj = (m["toks"] + m["ext_calls"] * EXT_TOKENS).mean()
        saving = (e3_adj - tok_adj) / e3_adj * 100
        calls = (m["ext_calls"] + m["rank_calls"] * (m["toks"] > 0)).mean()
        print(
            f"{name:44s} | {k/n:5.1%} | [{lo:5.1%}, {hi:5.1%}] | {tok_rank:13.0f} | "
            f"{tok_adj:15.0f} | {saving:+6.1f}% | {calls:9.2f}"
        )
        out.append(
            {
                "model": name,
                "top1_acc": round(k / n * 100, 1),
                "ci_low": round(lo * 100, 1),
                "ci_high": round(hi * 100, 1),
                "avg_tokens_ranker": round(float(tok_rank)),
                "avg_tokens_with_extractor": round(float(tok_adj)),
                "token_savings_pct_with_extractor": round(float(saving), 1),
                "avg_llm_calls": round(float(calls), 2),
            }
        )

    print("\n--- Paired bootstrap (difference in accuracy, 95% CI) ---")
    pairs = [
        ("Cascade + Verifier", "Fixed E3"),
        ("Cascade + Verifier", "Fixed E2"),
        ("Trained Router", "Fixed E3"),
        ("Trained Router", "Fixed E2"),
        ("Trained Router", "Cascade + Verifier"),
        ("Cascade + Verifier", "Filter only (extracted constraints)"),
    ]
    for a, b in pairs:
        d, lo, hi = paired_bootstrap(M[a]["correct"], M[b]["correct"])
        sig = "significant" if (lo > 0 or hi < 0) else "NOT significant"
        print(f"{a:28s} - {b:38s}: {d:+.3f}  [{lo:+.3f}, {hi:+.3f}]  {sig}")

    print("\n--- Extractor audit ---")
    print(
        f"Test queries with extracted != GT constraints: {len(ext_mismatch)} / {n} "
        f"({len(ext_mismatch)/n:.1%}). On these, the Verifier checks the WRONG "
        f"constraints and may report VALID incorrectly."
    )
    print(f"qids: {ext_mismatch[:20]}{' ...' if len(ext_mismatch) > 20 else ''}")
    print(
        "\nNote: Cascade here uses GT-validity of E1's top-1 as the verifier signal "
        "(equal to the real Verifier only when the extractor is error-free)."
    )

    with open("data/final_evaluation_v2.json", "w", encoding="utf-8") as f:
        json.dump(
            {"n_test": n, "extractor_tokens": EXT_TOKENS, "rows": out,
             "extractor_mismatch_qids": ext_mismatch},
            f, indent=2,
        )
    print("\nSaved -> data/final_evaluation_v2.json")


if __name__ == "__main__":
    main()
