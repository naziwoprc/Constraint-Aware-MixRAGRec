import json, collections
from extractor import extract

qs = [json.loads(l) for l in open("data/queries_para.jsonl")]
fields = ["max_price", "min_ram_gb", "min_storage_gb", "brand"]


def same(g, p, f):
    if g is None or p is None:
        return g is None and p is None
    return str(g).lower() == str(p).lower() if f == "brand" else float(g) == float(p)


by_style = collections.defaultdict(lambda: [0, 0])
errors, halluc, parse_fail = [], 0, 0

for q in qs:
    pred, raw = extract(q["query"])
    if pred is None:
        parse_fail += 1
        by_style[q["style"]][1] += 1
        errors.append((q["query"], "PARSE", raw[:100]))
        continue
    good = True
    for f in fields:
        g, p = q["constraints"].get(f), getattr(pred.hard, f)
        if not same(g, p, f):
            good = False
            errors.append((q["query"], f, f"gt={g} pred={p}"))
        if g is None and p is not None:
            halluc += 1
    if q["soft_pref"] != pred.soft_pref:
        good = False
        errors.append(
            (q["query"], "soft_pref", f"gt={q['soft_pref']} pred={pred.soft_pref}")
        )
    by_style[q["style"]][0] += good
    by_style[q["style"]][1] += 1

n = len(qs)
print("parse failures:", parse_fail, "| hallucinated:", halluc)
print("full match:", f"{sum(v[0] for v in by_style.values())/n:.0%}")
for s, (a, b) in by_style.items():
    print(f"{a}/{b}  {s[:60]}")
print("\n--- errors (first 15) ---")
for e in errors[:15]:
    print(e)
