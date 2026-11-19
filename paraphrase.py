import json, random
from extractor import client, MODEL, HEADERS, cache, PROV

random.seed(7)
qs = [json.loads(l) for l in open("data/queries.jsonl")]
sample = random.sample(qs, 150)

STYLES = [
    "casual and chatty, like a text message to a friend",
    "terse keyword style, like a search box query, no full sentences",
    "long and wordy, with some backstory about why they need it",
    "use informal number wording (e.g. 'bucks', 'gigs', 'a terabyte', 'sixteen')",
    "slightly messy: lowercase, minor typos, run-on sentence",
    "formal and polite email style",
]


def rewrite(q, style):
    prompt = (
        "Rewrite the shopping request below in a different style. "
        "Keep EXACTLY the same requirements: same numbers, same brand, same meaning "
        "of 'under/at most' and 'at least'. Keep optional wishes optional "
        "(do not turn a wish into a requirement, do not drop it). "
        "Do not add new requirements. Keep it under 80 words. "
        "Output only the rewritten text.\n\n"
        f"Style: {style}\nRequest: {q}"
    )
    key = f"para2|{MODEL}|{PROV}|{style}|{q}"
    if key not in cache:
        r = client.chat.completions.create(
            model=MODEL,
            temperature=0.7,
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
            extra_headers=HEADERS,
        )
        ch = r.choices[0]
        cache[key] = (ch.message.content.strip(), ch.finish_reason)
    return cache[key]


skipped = 0
with open("data/queries_para.jsonl", "w") as f:
    for q in sample:
        style = random.choice(STYLES)
        text, finish = rewrite(q["query"], style)
        if finish == "length":  # قطع‌شده، کنار بذار
            skipped += 1
            continue
        q2 = dict(q)
        q2["query_original"] = q["query"]
        q2["query"] = text
        q2["style"] = style
        f.write(json.dumps(q2) + "\n")
print("done:", len(sample) - skipped, "| skipped (truncated):", skipped)
