import pandas as pd, json

lap = pd.read_parquet("data/laptops_clean.parquet")


def det(x):
    try:
        return json.loads(x)
    except Exception:
        return {}


def num(x):
    return None if pd.isna(x) else round(float(x), 2)


cards = {}
for _, r in lap.iterrows():
    d = det(r["details"])
    cards[int(r["item_id"])] = {
        "title": str(r["title"])[:150],
        "brand": r["brand"] if isinstance(r["brand"], str) else None,
        "price": num(r["price_num"]),
        "ram_gb": num(r["ram_gb"]),
        "storage_gb": num(r["storage_gb"]),
        "weight_kg": num(r["weight_kg"]),
        "screen_in": num(r["screen_in"]),
        "cpu": d.get("Processor"),
        "os": d.get("Operating System"),
    }

json.dump(cards, open("data/cards.json", "w"))
print("cards:", len(cards))
k = next(iter(cards))
print(k, cards[k])
