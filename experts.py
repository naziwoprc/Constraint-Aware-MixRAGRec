import json

cards = {int(k): v for k, v in json.load(open("data/cards.json")).items()}


def _f(v, unit="", pre=""):
    return "unknown" if v is None else f"{pre}{v:g}{unit}"


FIELDS = {
    "brand": lambda c: f"brand: {c['brand'] or 'unknown'}",
    "price": lambda c: f"price: {_f(c['price'], pre='$')}",
    "ram": lambda c: f"RAM: {_f(c['ram_gb'], ' GB')}",
    "storage": lambda c: f"storage: {_f(c['storage_gb'], ' GB')}",
    "weight": lambda c: f"weight: {_f(c['weight_kg'], ' kg')}",
    "screen": lambda c: f"screen: {_f(c['screen_in'], ' in')}",
    "cpu": lambda c: f"CPU: {c['cpu'] or 'unknown'}",
    "os": lambda c: f"OS: {c['os'] or 'unknown'}",
}


def relevant_fields(ext):
    """E2: فقط ویژگی‌هایی که Extractor از query بیرون کشیده"""
    f = []
    if ext is None:
        return list(FIELDS)
    h = ext.hard
    if h.max_price is not None:
        f.append("price")
    if h.min_ram_gb is not None:
        f.append("ram")
    if h.min_storage_gb is not None:
        f.append("storage")
    if h.brand is not None:
        f.append("brand")
    if ext.soft_pref == "light":
        f.append("weight")
    if ext.soft_pref in ("large_screen", "small_screen"):
        f.append("screen")
    return f


def render(item_id, expert, ext=None):
    c = cards[item_id]
    if expert == "E1":
        return c["title"]
    fields = list(FIELDS) if expert == "E3" else relevant_fields(ext)
    return c["title"] + " | " + " | ".join(FIELDS[k](c) for k in fields)
