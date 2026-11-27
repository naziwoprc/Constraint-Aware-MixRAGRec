"""
Evaluation v3: Cascade with the REAL Verifier + Filter->LLM-rerank baseline.

Prerequisites:
    python collect_top1.py          # data/top1_data.jsonl
    python filter_rerank.py         # data/filter_rerank.jsonl  (optional)

Run:
    python evaluate_v3.py
"""

import os
import json
import numpy as np

from evaluate_v2 import wilson, paired_bootstrap, satisfies, EXT_TOKENS
from experts import cards
from extractor import extract


def load_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def main():
    queries = {q["qid"]: q for q in load_jsonl("data/queries.jsonl")}
    top1 = {r["qid"]: r for r in load_jsonl("data/top1_data.jsonl") if r["split"] == "test"}
    router_rows = [r for r in load_jsonl("data/router_data.jsonl") if r["split"] == "test"]
    qids = [r["qid"] for r in router_rows]
    missing = [q for q in qids if q not in top1]
    if missing:
        raise SystemExit(f"top1_data.jsonl lacks {len(missing)} test qids; rerun collect_top1.py")
    n = len(qids)

    fr = {}
    if os.path.exists("data/filter_rerank.jsonl"):
        fr = {r["qid"]: r for r in load_jsonl("data/filter_rerank.jsonl")}

    M = {}

    def add(name, correct, toks, ext_calls, rank_calls):
        M[name] = {
            "correct": np.array(correct, float),
            "toks": np.array(toks, float),
            "ext_calls": np.array(ext_calls, float),
            "rank_calls": np.array(rank_calls, float),
        }

    # ---- fixed baselines (from top1_data, consistent with cascade) ----
    for e, ext_c in (("E2", 1), ("E3", 0)):
        add(
            f"Fixed {e}",
            [top1[q]["out"][e]["ok"] for q in qids],
            [top1[q]["out"][e]["toks"] for q in qids],
            [ext_c] * n,
            [1] * n,
        )

    # ---- cascade: oracle signal vs REAL verifier ----
    c_or, t_or, k_or = [], [], []
    c_re, t_re, k_re = [], [], []
    tp = fp = fn = tn = 0
    false_accepts = []

    for qid in qids:
        q = queries[qid]
        o = top1[qid]["out"]
        e1, e2 = o["E1"], o["E2"]

        # oracle signal (GT validity)
        if e1["ok"]:
            c_or.append(1); t_or.append(e1["toks"]); k_or.append(1)
        else:
            c_or.append(e2["ok"]); t_or.append(e1["toks"] + e2["toks"]); k_or.append(2)

        # real verifier: extracted constraints on E1's top-1
        ext, _ = extract(q["query"])
        if ext is None:
            ex = {}  # mirrors agent_pipeline: nothing to check -> VALID
        else:
            h = ext.hard
            ex = {
                "max_price": h.max_price,
                "min_ram_gb": h.min_ram_gb,
                "min_storage_gb": h.min_storage_gb,
                "brand": h.brand,
            }
        t1 = e1["top1"]
        v_valid = t1 is not None and satisfies(cards[t1], ex)

        if v_valid and e1["ok"]:
            tp += 1
        elif v_valid and not e1["ok"]:
            fp += 1
            false_accepts.append(qid)
        elif (not v_valid) and e1["ok"]:
            fn += 1
        else:
            tn += 1

        if v_valid:
            c_re.append(int(e1["ok"])); t_re.append(e1["toks"]); k_re.append(1)
        else:
            c_re.append(int(e2["ok"])); t_re.append(e1["toks"] + e2["toks"]); k_re.append(2)

    add("Cascade (oracle signal, as in v1/v2)", c_or, t_or, [1] * n, k_or)
    add("Cascade + REAL Verifier", c_re, t_re, [1] * n, k_re)

    # ---- filter -> LLM rerank ----
    if fr:
        add(
            "Filter -> LLM rerank (survivors only)",
            [fr[q]["ok"] for q in qids],
            [fr[q]["toks"] for q in qids],
            [1] * n,
            [1 if fr[q]["llm_called"] else 0 for q in qids],
        )

    # ---------------------------------------------------------- report
    e3 = (M["Fixed E3"]["toks"] + M["Fixed E3"]["ext_calls"] * EXT_TOKENS).mean()
    print("=" * 112)
    print(f"EVALUATION v3 — TEST (n={n}), extractor = {EXT_TOKENS} tok/call")
    print("=" * 112)
    print(f"{'Method':42s} | {'Acc':6s} | {'95% CI':15s} | {'Tok(+ext)':9s} | {'Saving':7s} | {'LLM calls':9s}")
    print("-" * 112)
    rows = []
    for name, m in M.items():
        k = int(m["correct"].sum())
        lo, hi = wilson(k, n)
        tok = (m["toks"] + m["ext_calls"] * EXT_TOKENS).mean()
        calls = (m["ext_calls"] + m["rank_calls"]).mean()
        sav = (e3 - tok) / e3 * 100
        print(f"{name:42s} | {k/n:5.1%} | [{lo:5.1%},{hi:5.1%}] | {tok:9.0f} | {sav:+6.1f}% | {calls:9.2f}")
        rows.append({"model": name, "top1_acc": round(k / n * 100, 1),
                     "ci_low": round(lo * 100, 1), "ci_high": round(hi * 100, 1),
                     "avg_tokens_with_extractor": round(float(tok)),
                     "token_savings_pct": round(float(sav), 1),
                     "avg_llm_calls": round(float(calls), 2)})

    print("\n--- Verifier confusion on E1 top-1 (positive = 'VALID') ---")
    print(f"TP={tp}  FP(false accept)={fp}  FN(false reject)={fn}  TN={tn}")
    print(f"false-accept qids: {false_accepts}")
    print("FP = extractor error let a wrong answer through; "
          "FN = needless escalation to E2 (costs tokens only).")

    print("\n--- Paired bootstrap (acc difference, 95% CI) ---")
    pairs = [
        ("Cascade + REAL Verifier", "Cascade (oracle signal, as in v1/v2)"),
        ("Cascade + REAL Verifier", "Fixed E3"),
        ("Cascade + REAL Verifier", "Fixed E2"),
    ]
    if fr:
        pairs += [
            ("Filter -> LLM rerank (survivors only)", "Cascade + REAL Verifier"),
            ("Filter -> LLM rerank (survivors only)", "Fixed E3"),
        ]
    for a, b in pairs:
        d, lo, hi = paired_bootstrap(M[a]["correct"], M[b]["correct"])
        sig = "significant" if (lo > 0 or hi < 0) else "NOT significant"
        print(f"{a:40s} - {b:38s}: {d:+.3f} [{lo:+.3f},{hi:+.3f}] {sig}")

    with open("data/final_evaluation_v3.json", "w", encoding="utf-8") as f:
        json.dump({"n_test": n, "rows": rows,
                   "verifier_confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
                   "false_accept_qids": false_accepts}, f, indent=2)
    print("\nSaved -> data/final_evaluation_v3.json")


if __name__ == "__main__":
    main()
