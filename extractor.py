import os, json, re
from typing import Optional, Literal
from pydantic import BaseModel
from openai import OpenAI
from diskcache import Cache
from tenacity import retry, stop_after_attempt, wait_exponential

from dotenv import load_dotenv

load_dotenv()


PROV = os.environ.get("NANOGPT_PROVIDER")
HEADERS = {"X-Provider": PROV} if PROV else {}
client = OpenAI(
    base_url="https://nano-gpt.com/api/v1",
    api_key=os.environ["NANOGPT_API_KEY"],
)
MODEL = "deepseek/deepseek-v4.1-flash-1017"
cache = Cache("cache/llm")


class Hard(BaseModel):
    max_price: Optional[float] = None
    min_ram_gb: Optional[float] = None
    min_storage_gb: Optional[float] = None
    brand: Optional[str] = None


class Extraction(BaseModel):
    hard: Hard
    soft_pref: Optional[Literal["light", "large_screen", "small_screen"]] = None


SYSTEM = """You extract shopping constraints for laptops from a user request.
Return ONLY a JSON object, no explanation, with this exact shape:
{"hard": {"max_price": number|null, "min_ram_gb": number|null,
          "min_storage_gb": number|null, "brand": string|null},
 "soft_pref": "light"|"large_screen"|"small_screen"|null}

Rules:
- hard = requirements the user states as mandatory (under/at most/at least/must/from brand X).
- soft_pref = wishes marked as optional ("preferably", "would be nice", "ideally").
  Use "light" for lightweight/portable, "large_screen" for bigger screen,
  "small_screen" for compact screen.
- Convert storage to GB (1 TB = 1024 GB). Price is in USD, numbers only.
- If something is not mentioned, use null. Never invent constraints.

Example 1
Request: I need a laptop under $700 with at least 16 GB RAM from Dell. Preferably lightweight.
Output: {"hard": {"max_price": 700, "min_ram_gb": 16, "min_storage_gb": null, "brand": "Dell"}, "soft_pref": "light"}

Example 2
Request: Looking for a laptop with at least 512 GB storage and at least 8 GB RAM, no more than $900.
Output: {"hard": {"max_price": 900, "min_ram_gb": 8, "min_storage_gb": 512, "brand": null}, "soft_pref": null}
"""


@retry(stop=stop_after_attempt(4), wait=wait_exponential(min=2, max=30))
def _call(text):
    r = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        max_tokens=200,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Request: {text}\nOutput:"},
        ],
        extra_headers=HEADERS,
    )
    return r.choices[0].message.content


def extract(text):
    key = f"{MODEL}|{PROV}|{text}"
    if key not in cache:
        cache[key] = _call(text)
    raw = cache[key]
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        return Extraction.model_validate(json.loads(m.group(0))), raw
    except Exception:
        return None, raw
