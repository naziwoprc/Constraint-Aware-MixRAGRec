"""
FastAPI Backend for Constraint-Aware MixRAGRec 3D Studio
Run: python -m uvicorn api:app --reload --port 8000
"""

import hashlib
import json
import os
import random
import re
import time
from typing import List, Optional, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent_pipeline import build_mixragrec_graph, run_pipeline
from experts import cards
from extractor import extract
from pilot import qs

app = FastAPI(title="Constraint-Aware MixRAGRec API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

QS_BY_ID = {q["qid"]: q for q in qs}
QS_BY_TEXT = {q["query"].strip().lower(): q for q in qs}


class RunRequest(BaseModel):
    query: str
    mode: str = "cascade"  # "cascade" or "predictive"
    qid: Optional[int] = None


def count_exact_matches(hard: dict) -> int:
    """تعداد لپ‌تاپ‌هایی در کل کاتالوگ که تمام قیدهای سخت را ارضا می‌کنند."""
    if not hard:
        return len(cards)
    matches = 0
    for c in cards.values():
        if hard.get("max_price") is not None and (
            c["price"] is None or c["price"] > hard["max_price"]
        ):
            continue
        if hard.get("min_ram_gb") is not None and (
            c["ram_gb"] is None or c["ram_gb"] < hard["min_ram_gb"]
        ):
            continue
        if hard.get("min_storage_gb") is not None and (
            c["storage_gb"] is None or c["storage_gb"] < hard["min_storage_gb"]
        ):
            continue
        if hard.get("brand") is not None and (
            (c["brand"] or "").lower() != hard["brand"].lower()
        ):
            continue
        matches += 1
    return matches


def build_dynamic_pool(query: str, pool_size: int = 20) -> List[int]:
    """برای کوئری‌های دلخواه کاربر، یک استخر ۲۰تایی واقعی و تکرارپذیر از cards.json می‌سازد."""
    seed = int(hashlib.md5(query.strip().lower().encode("utf-8")).hexdigest(), 16)
    rng = random.Random(seed)
    ext, _ = extract(query)
    all_ids = list(cards.keys())
    if ext is None:
        return rng.sample(all_ids, min(pool_size, len(all_ids)))

    h = ext.hard
    ok, near, far = [], [], []
    for iid, c in cards.items():
        v = 0
        if h.max_price is not None and (c["price"] is None or c["price"] > h.max_price):
            v += 1
        if h.min_ram_gb is not None and (
            c["ram_gb"] is None or c["ram_gb"] < h.min_ram_gb
        ):
            v += 1
        if h.min_storage_gb is not None and (
            c["storage_gb"] is None or c["storage_gb"] < h.min_storage_gb
        ):
            v += 1
        if h.brand is not None and (c["brand"] or "").lower() != h.brand.lower():
            v += 1

        if v == 0:
            ok.append(iid)
        elif v == 1:
            near.append(iid)
        else:
            far.append(iid)

    pool = (
        rng.sample(ok, min(5, len(ok)))
        + rng.sample(near, min(10, len(near)))
        + rng.sample(far, min(5, len(far)))
    )
    remaining = [i for i in all_ids if i not in pool]
    if len(pool) < pool_size:
        pool += rng.sample(remaining, min(pool_size - len(pool), len(remaining)))
    rng.shuffle(pool)
    return pool[:pool_size]


def _resolve_pool_and_qid(req: RunRequest) -> Tuple[List[int], Optional[int]]:
    """استخر کاندیدا و شناسه کوئری را بر اساس qid، تطابق متن در دیتاست یا ساخت استخر پویا تعیین می‌کند."""
    if req.qid is not None and req.qid in QS_BY_ID:
        return QS_BY_ID[req.qid]["pool"], req.qid
    matched = QS_BY_TEXT.get(req.query.strip().lower())
    if matched is not None:
        return matched["pool"], matched["qid"]
    return build_dynamic_pool(req.query), None


def parse_tokens_from_history(history: List[str]) -> List[int]:
    """توکن مصرفی هر نوبت فراخوانی Recommender را از history استخراج می‌کند."""
    tokens = []
    for line in history:
        m = re.search(r"Prompt Tokens:\s*(\d+)", line)
        if m:
            tokens.append(int(m.group(1)))
    return tokens


@app.get("/")
def serve_index():
    index_path = os.path.join("static", "index.html")
    if not os.path.exists(index_path):
        return {"message": "API is running. Put your UI in static/index.html"}
    return FileResponse(index_path)


@app.get("/api/presets")
def get_presets():
    """چند کوئری نمونه از دیتاست واقعی برای تست سریع در رابط کاربری."""
    sample_qids = [0, 3, 5, 11, 15]
    presets = []
    for qid in sample_qids:
        if qid in QS_BY_ID:
            q = QS_BY_ID[qid]
            presets.append(
                {
                    "qid": q["qid"],
                    "query": q["query"],
                    "constraints": q["constraints"],
                    "soft_pref": q["soft_pref"],
                }
            )
    return presets


@app.get("/api/benchmarks")
def get_benchmarks():
    """خواندن نتایج واقعی بنچمارک از data/final_evaluation.json."""
    path = "data/final_evaluation.json"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="final_evaluation.json not found")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.post("/api/run")
def run_agent_pipeline(req: RunRequest):
    """اجرای واقعی پایپ‌لاین ۵ ایجنتی LangGraph."""
    if req.mode not in ("cascade", "predictive"):
        raise HTTPException(
            status_code=400, detail="mode must be 'cascade' or 'predictive'"
        )

    pool, qid = _resolve_pool_and_qid(req)

    res = run_pipeline(req.query, pool, qid=qid, mode=req.mode)
    turn_tokens = parse_tokens_from_history(res["history"])
    total_tokens = sum(turn_tokens) if turn_tokens else res["prompt_tokens"]
    hard_c = res.get("hard_constraints") or {}
    exact_matches = count_exact_matches(hard_c)

    ranked_items = [
        {"item_id": iid, **cards[iid]}
        for iid in res.get("ranked_item_ids", [])
        if iid in cards
    ]

    return {
        "query": res["query"],
        "mode": res["mode"],
        "hard_constraints": hard_c,
        "soft_pref": res.get("soft_pref"),
        "exact_matches_in_catalog": exact_matches,
        "is_infeasible": exact_matches == 0,
        "selected_expert": res["selected_expert"],
        "router_prob": res["router_prob"],
        "routing_reason": res["routing_reason"],
        "cascade_triggered": res.get("cascade_count", 0) > 0,
        "turn_tokens": turn_tokens,
        "total_tokens": total_tokens,
        "is_valid": res["is_valid"],
        "violations": res.get("violations", []),
        "verifier_feedback": res.get("verifier_feedback", ""),
        "top_item": res.get("top_item"),
        "ranked_items": ranked_items,
        "history": res.get("history", []),
    }


@app.post("/api/stream")
def stream_agent_pipeline(req: RunRequest):
    """اجرای لحظه‌ای (Real-Time) پایپ‌لاین LangGraph با استریم SSE."""
    if req.mode not in ("cascade", "predictive"):
        raise HTTPException(
            status_code=400, detail="mode must be 'cascade' or 'predictive'"
        )

    pool, qid = _resolve_pool_and_qid(req)

    graph = build_mixragrec_graph()
    initial_state = {
        "qid": qid,
        "query": req.query,
        "candidate_pool": pool,
        "mode": req.mode,
        "ext_obj": None,
        "hard_constraints": None,
        "soft_pref": None,
        "selected_expert": "E1",
        "router_prob": 0.0,
        "routing_reason": "",
        "rendered_candidates": [],
        "ranked_item_ids": [],
        "top_item": None,
        "prompt_tokens": 0,
        "is_valid": False,
        "violations": [],
        "verifier_feedback": "",
        "cascade_count": 0,
        "history": [],
    }

    def event_generator():
        t_start = time.perf_counter()
        last_t = t_start

        for step in graph.stream(initial_state):
            now = time.perf_counter()
            step_ms = round((now - last_t) * 1000)
            last_t = now

            for node_name, output in step.items():
                payload = {
                    "node": node_name,
                    "step_ms": step_ms,
                    "elapsed_ms": round((now - t_start) * 1000),
                }

                if node_name == "extractor":
                    hard_c = output.get("hard_constraints") or {}
                    exact_matches = count_exact_matches(hard_c)
                    payload["hard_constraints"] = hard_c
                    payload["soft_pref"] = output.get("soft_pref")
                    payload["exact_matches_in_catalog"] = exact_matches
                    payload["is_infeasible"] = exact_matches == 0
                elif node_name == "router":
                    payload["selected_expert"] = output.get("selected_expert")
                    payload["router_prob"] = output.get("router_prob")
                    payload["routing_reason"] = output.get("routing_reason")
                elif node_name == "retriever":
                    payload["candidate_count"] = len(
                        output.get("rendered_candidates", [])
                    )
                elif node_name == "recommender":
                    payload["prompt_tokens"] = output.get("prompt_tokens", 0)
                    payload["top_item"] = output.get("top_item")
                    payload["ranked_items"] = [
                        {"item_id": iid, **cards[iid]}
                        for iid in output.get("ranked_item_ids", [])
                        if iid in cards
                    ]
                elif node_name == "verifier":
                    payload["is_valid"] = output.get("is_valid")
                    payload["violations"] = output.get("violations", [])
                    payload["verifier_feedback"] = output.get("verifier_feedback")
                elif node_name == "fallback_handler":
                    payload["cascade_count"] = output.get("cascade_count", 1)

                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

        yield f"data: {json.dumps({'node': 'done', 'elapsed_ms': round((time.perf_counter() - t_start) * 1000)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
