"""
Collect the actual ranked item ids (not just ok/toks) for every expert.
Needed to simulate the REAL Verifier in the cascade.

Usage:
    python collect_top1.py            # test split only (default)
    python collect_top1.py test dev   # several splits
    python collect_top1.py all

Results are read from the LLM cache when possible (same cache key as
pilot.run), so this normally makes NO new API calls. A cache-miss counter
is printed so you know if anything was actually re-queried.
"""

import sys
import json
from pilot import run, qs, split, cache, MODEL, PROV

ORDER = ["E1", "E2", "E3"]
OUT = "data/top1_data.jsonl"


def main():
    args = sys.argv[1:] or ["test"]
    wanted = {"train", "dev", "test"} if "all" in args else set(args)

    misses = 0
    n_rows = 0
    with open(OUT, "w", encoding="utf-8") as f:
        for q in qs:
            s = split[str(q["qid"])]
            if s not in wanted:
                continue
            row = {"qid": q["qid"], "split": s, "out": {}}
            for e in ORDER:
                key = f"rec|{MODEL}|{PROV}|{e}|{q['qid']}"
                if key not in cache:
                    misses += 1
                ranked, ptoks, fin = run(q, e)
                row["out"][e] = {
                    "ranked": ranked,
                    "top1": ranked[0] if ranked else None,
                    "ok": bool(ranked) and ranked[0] in q["valid_in_pool"],
                    "toks": ptoks,
                }
            f.write(json.dumps(row) + "\n")
            n_rows += 1
            if n_rows % 25 == 0:
                print(f"  {n_rows} queries...", flush=True)

    print(f"Saved {n_rows} rows -> {OUT}")
    print(f"Cache misses (new API calls): {misses}")


if __name__ == "__main__":
    main()
