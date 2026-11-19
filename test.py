import os, time
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(base_url="https://nano-gpt.com/api/v1", api_key=os.environ["NANOGPT_API_KEY"])
MODEL = "deepseek/deepseek-v4.1-flash-1017"
PROVIDERS = [None, "deepinfra", "cloudflare", "discretestack", "coralbricks-standard"]
PROMPT = 'Extract JSON {"max_price":number,"min_ram_gb":number} from: I need a laptop under $700 with at least 16 GB RAM.'

for prov in PROVIDERS:
    times, tok = [], 0
    for _ in range(3):
        t = time.time()
        try:
            r = client.chat.completions.create(
                model=MODEL, temperature=0, max_tokens=200,
                messages=[{"role": "user", "content": PROMPT}],
                extra_headers={"X-Provider": prov} if prov else {})
            times.append(time.time() - t)
            tok = r.usage.completion_tokens
            out = r.choices[0].message.content
        except Exception as e:
            times.append(None); out = f"ERR {str(e)[:120]}"
    ok = [x for x in times if x]
    avg = sum(ok) / len(ok) if ok else float("nan")
    print(f"{str(prov):22s} avg={avg:.2f}s out_tokens={tok} | {out[:80]!r}")