import sys, json, re, statistics, os
from pilot import run, qs, split
from extractor import extract
from experts import cards

ORDER = ["E1", "E2", "E3"]
test_res = json.load(open("data/test_results.json"))

limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
queries_to_run = qs[:limit] if limit else qs
out_file = "data/router_data_sample.jsonl" if limit else "data/router_data.jsonl"


def feats(q):
    ext, _ = extract(q["query"])
    titles = [cards[i]["title"] for i in q["pool"]]
    f = {
        "frac_title_gb": statistics.mean(
            1 if re.search(r"\d+\s*GB", t, re.I) else 0 for t in titles
        )
    }
    if ext is not None:
        h = ext.hard
        f.update(
            max_price=h.max_price,
            min_ram_gb=h.min_ram_gb,
            min_storage_gb=h.min_storage_gb,
            has_brand=h.brand is not None,
            soft=ext.soft_pref,
            n_hard=sum(
                x is not None
                for x in (h.max_price, h.min_ram_gb, h.min_storage_gb, h.brand)
            ),
        )
    else:
        f.update(
            max_price=None,
            min_ram_gb=None,
            min_storage_gb=None,
            has_brand=False,
            soft=None,
            n_hard=0,
        )
    return f


already_done = set()
if os.path.exists(out_file) and not limit:
    for line in open(out_file, "r", encoding="utf-8"):
        try:
            already_done.add(json.loads(line)["qid"])
        except Exception:
            pass

print(f"Target: {len(queries_to_run)} queries | Already collected in file: {len(already_done)}")

with open(out_file, "a" if not limit else "w", encoding="utf-8") as f:
    for i, q in enumerate(queries_to_run, 1):
        if q["qid"] in already_done:
            continue

        s = split[str(q["qid"])]
        if s == "test":
            outcomes = test_res[str(q["qid"])]
        else:
            outcomes = {}
            for e in ORDER:
                ranked, ptoks, fin = run(q, e)
                outcomes[e] = {
                    "ok": bool(ranked) and ranked[0] in q["valid_in_pool"],
                    "toks": ptoks,
                }

        row = {"qid": q["qid"], "split": s, "feats": feats(q), "out": outcomes}
        f.write(json.dumps(row) + "\n")
        f.flush()

        if i % 10 == 0 or i == len(queries_to_run):
            print(f"Progress: {i} / {len(queries_to_run)}", flush=True)

# Count final saved rows
total_saved = sum(1 for _ in open(out_file, "r", encoding="utf-8"))
print(f"Finished! Total saved in {out_file}: {total_saved}")
