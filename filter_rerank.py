"""
Baseline: deterministic hard-constraint filter -> LLM ranks ONLY the survivors.

Pipeline:
  Extractor -> filter pool with extracted hard constraints (pure Python)
            -> if 0 survivors: no answer (counted wrong, 0 tokens)
            -> if 1 survivor : return it (no LLM call)
            -> else          : LLM ranks survivors (E2-style rendering)

If the extractor fails to parse, no filtering is applied (full pool).

Usage:  python filter_rerank.py          # test split
        python filter_rerank.py dev test
Output: data/filter_rerank.jsonl
"""

import sys
import re
import json
import hashlib

from pilot import call, qs, split, cache, MODEL, PROV
from extractor import extract
from experts import cards, render
from evaluate_v2 import satisfies

OUT = "data/filter_rerank.jsonl"


def llm_rank(query, survivors, ext):
    lines = [f"{i+1}. {render(iid, 'E2', ext)}" for i, iid in enumerate(survivors)]
    msg = f"Request: {query}\n\nCandidates:\n" + "\n".join(lines)
    h = hashlib.md5(msg.encode("utf-8")).hexdigest()
    key = f"filterrerank|{MODEL}|{PROV}|{h}"
    if key not in cache:
        cache[key] = call(msg)
    raw, ptoks, fin = cache[key]

    m = re.search(r"\[[\d,\s]+\]", raw or "")
    try:
        nums = [n for n in json.loads(m.group(0)) if 1 <= n <= len(survivors)]
        ranked = [survivors[n - 1] for n in dict.fromkeys(nums)]
    except Exception:
        ranked = []
    return ranked, ptoks


def main():
    wanted = set(sys.argv[1:]) or {"test"}
    n = 0
    with open(OUT, "w", encoding="utf-8") as f:
        for q in qs:
            s = split[str(q["qid"])]
            if s not in wanted:
                continue

            ext, _ = extract(q["query"])
            pool = q["pool"]
            if ext is None:
                survivors = list(pool)
            else:
                h = ext.hard
                ex = {
                    "max_price": h.max_price,
                    "min_ram_gb": h.min_ram_gb,
                    "min_storage_gb": h.min_storage_gb,
                    "brand": h.brand,
                }
                survivors = [i for i in pool if satisfies(cards[i], ex)]

            ptoks, llm_called = 0, False
            if not survivors:
                top1 = None
            elif len(survivors) == 1:
                top1 = survivors[0]
            else:
                ranked, ptoks = llm_rank(q["query"], survivors, ext)
                llm_called = True
                top1 = ranked[0] if ranked else None

            row = {
                "qid": q["qid"],
                "split": s,
                "n_survivors": len(survivors),
                "top1": top1,
                "ok": top1 is not None and top1 in q["valid_in_pool"],
                "toks": ptoks,
                "llm_called": llm_called,
            }
            f.write(json.dumps(row) + "\n")
            f.flush()
            n += 1
            if n % 25 == 0:
                print(f"  {n} queries...", flush=True)

    print(f"Saved {n} rows -> {OUT}")


if __name__ == "__main__":
    main()
