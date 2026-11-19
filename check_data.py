import pandas as pd, random, json

random.seed(42)

lap = pd.read_parquet("data/laptops.parquet")
lap = lap[
    (lap.price_num >= 100)
    & lap.ram_gb.notna()
    & lap.weight_kg.between(0.7, 5)
    & lap.screen_in.notna()
].copy()
lap = lap.reset_index(drop=True)
lap["item_id"] = lap.index
print("clean laptops:", len(lap))
lap.to_parquet("data/laptops_clean.parquet")

BRANDS = [b for b, n in lap.brand.value_counts().items() if n >= 20]


def sample_constraints():
    c = {
        "max_price": random.choice([400, 500, 600, 700, 800, 1000, 1200]),
        "min_ram_gb": random.choice([8, 16, 32]),
    }
    if random.random() < 0.4:
        c["brand"] = random.choice(BRANDS)
    if random.random() < 0.3:
        c["min_storage_gb"] = random.choice([256, 512, 1024])
    soft = random.choice(["light", "large_screen", "small_screen", None])
    return c, soft


def violations(df, c):
    v = (df.price_num > c["max_price"]).astype(int) + (
        df.ram_gb < c["min_ram_gb"]
    ).astype(int)
    if "brand" in c:
        v = v + (df.brand.str.lower() != c["brand"].lower()).astype(int)
    if "min_storage_gb" in c:
        v = v + (~(df.storage_gb >= c["min_storage_gb"])).astype(int)
    return v


def to_text(c, soft):
    s = f"I need a laptop under ${c['max_price']} with at least {c['min_ram_gb']} GB RAM"
    if "brand" in c:
        s += f" from {c['brand']}"
    if "min_storage_gb" in c:
        s += f" and at least {c['min_storage_gb']} GB storage"
    s += "."
    s += {
        "light": " Preferably lightweight.",
        "large_screen": " A larger screen would be nice.",
        "small_screen": " A compact screen is preferred.",
        None: "",
    }[soft]
    return s


queries, seen, tries = [], set(), 0
while len(queries) < 600 and tries < 30000:
    tries += 1
    c, soft = sample_constraints()
    key = json.dumps(c, sort_keys=True) + str(soft)
    if key in seen:
        continue
    v = violations(lap, c)
    ok, near, far = lap[v == 0], lap[v == 1], lap[v >= 2]
    if not (5 <= len(ok) <= 60) or len(near) < 10 or len(far) < 5:
        continue
    seen.add(key)
    pool = (
        ok.sample(5, random_state=tries).item_id.tolist()
        + near.sample(10, random_state=tries).item_id.tolist()
        + far.sample(5, random_state=tries).item_id.tolist()
    )
    valid_in_pool = pool[:5]
    random.shuffle(pool)
    queries.append(
        {
            "qid": len(queries),
            "query": to_text(c, soft),
            "constraints": c,
            "soft_pref": soft,
            "n_valid_total": len(ok),
            "pool": pool,
            "valid_in_pool": valid_in_pool,
        }
    )

with open("data/queries.jsonl", "w") as f:
    for q in queries:
        f.write(json.dumps(q) + "\n")

print("queries:", len(queries), "after", tries, "tries")
print(queries[0]["query"], "| valid total:", queries[0]["n_valid_total"])
print(queries[1]["query"], "| valid total:", queries[1]["n_valid_total"])
