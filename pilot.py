import json, re, time
from tenacity import retry, stop_after_attempt, wait_exponential
from extractor import client, MODEL, HEADERS, cache, PROV, extract
from experts import render

split = json.load(open("data/split.json"))
qs = [json.loads(l) for l in open("data/queries.jsonl")]
train = [q for q in qs if split[str(q["qid"])] == "train"][:20]

SYSTEM = """You are a laptop shopping assistant. The user states requirements.
From the numbered candidates, choose the 5 best and rank them.
Rules:
- A candidate must satisfy ALL hard requirements (price limit, minimum RAM,
  minimum storage, brand). Optional wishes ("preferably", "would be nice")
  may only be used to order candidates that already satisfy the requirements.
- Use ONLY the information shown for each candidate. If a spec needed to verify
  a requirement is not shown, you cannot verify it.
Return ONLY JSON: {"ranking": [n1, n2, n3, n4, n5]} using candidate numbers."""


@retry(stop=stop_after_attempt(7), wait=wait_exponential(min=2, max=60))
def call(user_msg):
    r = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        max_tokens=300,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        extra_headers=HEADERS,
    )
    ch = r.choices[0]
    return ch.message.content, r.usage.prompt_tokens, ch.finish_reason


def run(q, expert):
    ext, _ = extract(q["query"])
    lines = [f"{i+1}. {render(iid, expert, ext)}" for i, iid in enumerate(q["pool"])]
    msg = f"Request: {q['query']}\n\nCandidates:\n" + "\n".join(lines)
    key = f"rec|{MODEL}|{PROV}|{expert}|{q['qid']}"
    if key not in cache:
        cache[key] = call(msg)
    raw, ptoks, fin = cache[key]
    m = re.search(r"\[[\d,\s]+\]", raw or "")
    try:
        nums = [n for n in json.loads(m.group(0)) if 1 <= n <= len(q["pool"])]
        ranked = [q["pool"][n - 1] for n in dict.fromkeys(nums)]
    except Exception:
        return None, ptoks, fin
    return ranked[:5], ptoks, fin


if __name__ == "__main__":
    valid = lambda q: set(q["valid_in_pool"])

    print(f"{len(train)} train queries | random top-1 baseline = 25%\n")
    for expert in ["E1", "E2", "E3"]:
        top1 = p5 = fails = trunc = 0
        toks = []
        for q in train:
            ranked, ptoks, fin = run(q, expert)
            toks.append(ptoks)
            trunc += fin == "length"
            if not ranked:
                fails += 1
                continue
            top1 += ranked[0] in valid(q)
            p5 += len(set(ranked) & valid(q)) / 5
        n = len(train)
        print(
            f"{expert}: top1={top1/n:.0%}  P@5={p5/n:.0%}  "
            f"avg_prompt_tokens={sum(toks)/n:.0f}  parse_fail={fails}  truncated={trunc}"
        )
