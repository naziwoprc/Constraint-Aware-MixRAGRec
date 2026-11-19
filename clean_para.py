import json
from extractor import extract

qs = [json.loads(l) for l in open("data/queries_para.jsonl")]
keep, dropped = [], []
for q in qs:
    pred, _ = extract(q["query"])
    if pred and q["soft_pref"] is None and pred.soft_pref is not None:
        dropped.append(q)
    else:
        keep.append(q)

for q in dropped:
    print(q["qid"], "|", q["query"][:160], "\n   original:", q["query_original"], "\n")

with open("data/queries_para_clean.jsonl", "w") as f:
    for q in keep:
        f.write(json.dumps(q) + "\n")
print("kept:", len(keep), "dropped:", len(dropped))
