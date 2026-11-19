import json, collections
from pilot import run, qs, split
from experts import cards

N = 100
train = [q for q in qs if split[str(q["qid"])] == "train"][:N]
ORDER = ["E1", "E2", "E3"]


def violated(q, iid):
    c, k = q["constraints"], cards[iid]
    v = []
    if k["price"] is None or k["price"] > c["max_price"]:
        v.append("price")
    if k["ram_gb"] is None or k["ram_gb"] < c["min_ram_gb"]:
        v.append("ram")
    if "brand" in c and (k["brand"] or "").lower() != c["brand"].lower():
        v.append("brand")
    if "min_storage_gb" in c and not (
        k["storage_gb"] is not None and k["storage_gb"] >= c["min_storage_gb"]
    ):
        v.append("storage")
    return v


res, wrong = {}, collections.Counter()
for q in train:
    row = {}
    for e in ORDER:
        ranked, ptoks, fin = run(q, e)
        ok = bool(ranked) and ranked[0] in q["valid_in_pool"]
        row[e] = {"ok": ok, "toks": ptoks}
        if e == "E1" and ranked and not ok:
            for v in violated(q, ranked[0]):
                wrong[v] += 1
    res[q["qid"]] = row
json.dump(res, open("data/pilot_results.json", "w"))

n = len(res)
print(f"{n} queries")
for e in ORDER:
    acc = sum(r[e]["ok"] for r in res.values()) / n
    tk = sum(r[e]["toks"] for r in res.values()) / n
    print(f"{e}: top1={acc:.0%}  avg_tokens={tk:.0f}")

print("\nE1 wrong top-1 violated:", dict(wrong))

cat = collections.Counter()
oracle_toks = 0
for r in res.values():
    for e in ORDER:
        if r[e]["ok"]:
            cat[f"cheapest_ok={e}"] += 1
            oracle_toks += r[e]["toks"]
            break
    else:
        cat["none_ok"] += 1
        oracle_toks += r["E1"]["toks"]
print("\n", dict(cat))
print(f"oracle router avg tokens: {oracle_toks/n:.0f}")
