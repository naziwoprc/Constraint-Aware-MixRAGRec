import json, random, collections

qs = [json.loads(l) for l in open("data/queries.jsonl")]
groups = collections.defaultdict(list)
for q in qs:
    groups[json.dumps(q["constraints"], sort_keys=True)].append(q["qid"])
keys = list(groups)
random.Random(42).shuffle(keys)

split, n = {}, 0
for k in keys:
    s = "train" if n < 300 else "dev" if n < 400 else "test"
    for qid in groups[k]:
        split[qid] = s
    n += len(groups[k])
json.dump(split, open("data/split.json", "w"))
print(collections.Counter(split.values()))
