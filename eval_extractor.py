import json, time
from extractor import extract

qs = [json.loads(l) for l in open("data/queries.jsonl")][:50]  # اول ۵۰ تا

fields = ["max_price", "min_ram_gb", "min_storage_gb", "brand"]
ok = {f: 0 for f in fields + ["soft_pref"]}
halluc = parse_fail = full = 0

for q in qs:
    pred, raw = extract(q["query"])

    if pred is None:
        parse_fail += 1
        continue
    gt, good = q["constraints"], True
    for f in fields:
        g, p = gt.get(f), getattr(pred.hard, f)
        same = (g is None and p is None) or (
            g is not None and p is not None and str(g).lower() == str(p).lower()
            if f == "brand"
            else g == p
        )
        ok[f] += same
        good &= same
        if g is None and p is not None:
            halluc += 1
    s = q["soft_pref"] == pred.soft_pref
    ok["soft_pref"] += s
    good &= s
    full += good

n = len(qs)
print("parse failures:", parse_fail)
for f, v in ok.items():
    print(f"{f:16s} {v/n:.0%}")
print("hallucinated hard constraints:", halluc)
print("full match:", f"{full/n:.0%}")
