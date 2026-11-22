"""
MixRAGRec Multi-Agent Workflow Pipeline with LangGraph
=====================================================
Orchestrates the cooperative multi-agent decision process:
1. Constraint Extractor Agent  (LLM-based)
2. MoE Router Agent            (ML / Policy-based)
3. Knowledge Retriever Agent   (Granularity-aware Schema Pruning)
4. Recommendation Agent        (LLM Ranker)
5. Evidence Verifier Agent     (Deterministic Guardrail)
"""

import sys, os, json, re, statistics, joblib
import numpy as np
from typing import TypedDict, List, Optional, Dict, Any

from langgraph.graph import StateGraph, START, END

# Import existing modules
from extractor import extract
from experts import cards, render
from pilot import call, MODEL, PROV, cache
from train_router import extract_features

# Load trained router model
ROUTER_BUNDLE_PATH = "data/router_model.joblib"
router_bundle = None
if os.path.exists(ROUTER_BUNDLE_PATH):
    router_bundle = joblib.load(ROUTER_BUNDLE_PATH)


class AgentState(TypedDict):
    qid: Optional[int]
    query: str
    candidate_pool: List[int]
    mode: str  # "cascade" or "predictive"

    # Extractor outputs
    ext_obj: Optional[Any]
    hard_constraints: Optional[Dict[str, Any]]
    soft_pref: Optional[str]

    # Router outputs
    selected_expert: str  # E1, E2, E3
    router_prob: float
    routing_reason: str

    # Retriever outputs
    rendered_candidates: List[str]

    # Recommender outputs
    ranked_item_ids: List[int]
    top_item: Optional[Dict[str, Any]]
    prompt_tokens: int

    # Verifier outputs
    is_valid: bool
    violations: List[str]
    verifier_feedback: str

    # Execution tracking
    cascade_count: int
    history: List[str]


# -------------------------------------------------------------
# Agent 1: Constraint Extractor Agent
# -------------------------------------------------------------
def extractor_node(state: AgentState) -> Dict[str, Any]:
    query = state["query"]
    ext, raw = extract(query)

    hard = {}
    soft = None
    if ext is not None:
        h = ext.hard
        hard = {
            "max_price": h.max_price,
            "min_ram_gb": h.min_ram_gb,
            "min_storage_gb": h.min_storage_gb,
            "brand": h.brand,
        }
        soft = ext.soft_pref

    log_entry = (
        f"[Extractor Agent] Extracted Hard Constraints: {hard} | Soft Pref: {soft}"
    )
    history = state.get("history", []) + [log_entry]

    return {
        "ext_obj": ext,
        "hard_constraints": hard,
        "soft_pref": soft,
        "history": history,
    }


# -------------------------------------------------------------
# Agent 2: MoE Router Agent
# -------------------------------------------------------------
def router_node(state: AgentState) -> Dict[str, Any]:
    history = state.get("history", [])
    mode = state.get("mode", "cascade")
    cascade_count = state.get("cascade_count", 0)

    # If in Cascade mode and already failed E1 once, fallback to E2
    if mode == "cascade" and cascade_count > 0:
        log_entry = (
            "[Router Agent] Cascade fallback triggered -> Escalating to Expert E2"
        )
        return {
            "selected_expert": "E2",
            "router_prob": 0.0,
            "routing_reason": "Fallback from failed E1 verification",
            "history": history + [log_entry],
        }

    # If in Cascade mode on initial attempt: start with E1 (cheapest)
    if mode == "cascade":
        log_entry = "[Router Agent] Cascade Mode: starting with cheapest Expert E1 (Title only)"
        return {
            "selected_expert": "E1",
            "router_prob": 1.0,
            "routing_reason": "Cascade initial optimistic choice",
            "history": history + [log_entry],
        }

    # In Predictive mode: Use trained classifier
    titles = [cards[i]["title"] for i in state["candidate_pool"]]
    frac_title_gb = statistics.mean(
        1 if re.search(r"\d+\s*GB", t, re.I) else 0 for t in titles
    )

    h = state.get("hard_constraints", {})
    feat_dict = {
        "frac_title_gb": frac_title_gb,
        "max_price": h.get("max_price") or 1000.0,
        "min_ram_gb": h.get("min_ram_gb") or 8.0,
        "min_storage_gb": h.get("min_storage_gb"),
        "has_brand": h.get("brand") is not None,
        "soft": state.get("soft_pref"),
        "n_hard": sum(1 for v in h.values() if v is not None),
    }

    dummy_row = {"feats": feat_dict}
    X = np.array([extract_features(dummy_row)])

    if router_bundle is not None:
        clf = router_bundle["classifier"]
        scaler = router_bundle["scaler"]
        threshold = router_bundle["threshold"]
        X_s = scaler.transform(X)
        prob_e1_safe = float(clf.predict_proba(X_s)[0, 1])
        selected = "E1" if prob_e1_safe >= threshold else "E2"
        reason = f"P(E1 safe)={prob_e1_safe:.3f} vs threshold={threshold:.2f} -> {selected}"
    else:
        prob_e1_safe = 0.5
        selected = "E2"
        reason = "Default fallback (model not loaded)"

    log_entry = f"[Router Agent] Predictive Mode: {reason}"
    return {
        "selected_expert": selected,
        "router_prob": prob_e1_safe,
        "routing_reason": reason,
        "history": history + [log_entry],
    }


# -------------------------------------------------------------
# Agent 3: Knowledge Retriever Agent
# -------------------------------------------------------------
def retriever_node(state: AgentState) -> Dict[str, Any]:
    expert = state["selected_expert"]
    ext = state.get("ext_obj")
    pool = state["candidate_pool"]

    lines = [f"{i+1}. {render(iid, expert, ext)}" for i, iid in enumerate(pool)]
    log_entry = (
        f"[Retriever Agent] Retrieved {len(lines)} candidates using Expert {expert} "
        f"({'Title only' if expert=='E1' else 'Targeted specs'})"
    )

    return {
        "rendered_candidates": lines,
        "history": state.get("history", []) + [log_entry],
    }


# -------------------------------------------------------------
# Agent 4: Recommendation Agent
# -------------------------------------------------------------
def recommender_node(state: AgentState) -> Dict[str, Any]:
    query = state["query"]
    lines = state["rendered_candidates"]
    pool = state["candidate_pool"]
    expert = state["selected_expert"]

    msg = f"Request: {query}\n\nCandidates:\n" + "\n".join(lines)
    
    qid = state.get("qid")
    rec_key = f"rec|{MODEL}|{PROV}|{expert}|{qid}" if qid is not None else None
    
    if rec_key and rec_key in cache:
        raw, ptoks, fin = cache[rec_key]
    else:
        import hashlib
        msg_hash = hashlib.md5(msg.encode("utf-8")).hexdigest()
        key = f"pipeline|{MODEL}|{PROV}|{expert}|{msg_hash}"
        if key not in cache:
            cache[key] = call(msg)
        raw, ptoks, fin = cache[key]

    m = re.search(r"\[[\d,\s]+\]", raw or "")
    try:
        nums = [n for n in json.loads(m.group(0)) if 1 <= n <= len(pool)]
        ranked = [pool[n - 1] for n in dict.fromkeys(nums)]
    except Exception:
        ranked = pool[:5]

    top_id = ranked[0] if ranked else pool[0]
    top_item = cards.get(top_id)

    log_entry = (
        f"[Recommender Agent] Ranked top 5 items (Prompt Tokens: {ptoks}). "
        f"Top-1: [{top_id}] {top_item['title'][:60]}..."
    )

    return {
        "ranked_item_ids": ranked[:5],
        "top_item": top_item,
        "prompt_tokens": ptoks,
        "history": state.get("history", []) + [log_entry],
    }


# -------------------------------------------------------------
# Agent 5: Evidence Verifier Agent
# -------------------------------------------------------------
def verifier_node(state: AgentState) -> Dict[str, Any]:
    top_item = state.get("top_item")
    c = state.get("hard_constraints", {})

    violations = []
    if top_item:
        if c.get("max_price") and (
            top_item["price"] is None or top_item["price"] > c["max_price"]
        ):
            violations.append(
                f"Price (${top_item['price']}) > Max (${c['max_price']})"
            )
        if c.get("min_ram_gb") and (
            top_item["ram_gb"] is None or top_item["ram_gb"] < c["min_ram_gb"]
        ):
            violations.append(
                f"RAM ({top_item['ram_gb']}GB) < Min ({c['min_ram_gb']}GB)"
            )
        if (
            c.get("brand")
            and (top_item["brand"] or "").lower() != c["brand"].lower()
        ):
            violations.append(
                f"Brand ({top_item['brand']}) != Required ({c['brand']})"
            )
        if c.get("min_storage_gb") and not (
            top_item["storage_gb"] is not None
            and top_item["storage_gb"] >= c["min_storage_gb"]
        ):
            violations.append(
                f"Storage ({top_item['storage_gb']}GB) < Min ({c['min_storage_gb']}GB)"
            )

    is_valid = len(violations) == 0
    feedback = (
        "VALID: Top-1 candidate satisfies all hard constraints."
        if is_valid
        else f"INVALID: Violations: {', '.join(violations)}"
    )

    log_entry = f"[Verifier Agent] {feedback}"
    return {
        "is_valid": is_valid,
        "violations": violations,
        "verifier_feedback": feedback,
        "history": state.get("history", []) + [log_entry],
    }


# -------------------------------------------------------------
# Conditional Edge: Check if Cascade Fallback is Needed
# -------------------------------------------------------------
def check_verifier_condition(state: AgentState) -> str:
    # If valid, we are done
    if state["is_valid"]:
        return "success"

    # If invalid AND we are in cascade mode AND this was the first attempt (E1):
    if (
        state.get("mode") == "cascade"
        and state.get("selected_expert") == "E1"
        and state.get("cascade_count", 0) == 0
    ):
        return "fallback"

    return "end_with_warning"


def handle_fallback(state: AgentState) -> Dict[str, Any]:
    log_entry = "[Orchestrator] Verification failed on E1. Triggering Cascade to Expert E2."
    return {
        "cascade_count": state.get("cascade_count", 0) + 1,
        "history": state.get("history", []) + [log_entry],
    }


# -------------------------------------------------------------
# Build the LangGraph StateGraph
# -------------------------------------------------------------
def build_mixragrec_graph():
    builder = StateGraph(AgentState)

    # Add Nodes
    builder.add_node("extractor", extractor_node)
    builder.add_node("router", router_node)
    builder.add_node("retriever", retriever_node)
    builder.add_node("recommender", recommender_node)
    builder.add_node("verifier", verifier_node)
    builder.add_node("fallback_handler", handle_fallback)

    # Standard linear flow
    builder.add_edge(START, "extractor")
    builder.add_edge("extractor", "router")
    builder.add_edge("router", "retriever")
    builder.add_edge("retriever", "recommender")
    builder.add_edge("recommender", "verifier")

    # Conditional branching from Verifier
    builder.add_conditional_edges(
        "verifier",
        check_verifier_condition,
        {
            "success": END,
            "fallback": "fallback_handler",
            "end_with_warning": END,
        },
    )

    # From fallback handler, loop back to router (which picks E2)
    builder.add_edge("fallback_handler", "router")

    return builder.compile()


# -------------------------------------------------------------
# Demonstration / Execution Runner
# -------------------------------------------------------------
def run_pipeline(
    query: str, pool: List[int], qid: Optional[int] = None, mode: str = "cascade"
) -> AgentState:
    graph = build_mixragrec_graph()

    initial_state: AgentState = {
        "qid": qid,
        "query": query,
        "candidate_pool": pool,
        "mode": mode,
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

    result = graph.invoke(initial_state)
    return result


if __name__ == "__main__":
    from pilot import qs

    # Test 1: Query 0 (Predictive & Cascade - both solve with E1)
    q0 = qs[0]
    print("=" * 75)
    print("TEST 1: PREDICTIVE ROUTER ON QUERY 0")
    print(f"Query: \"{q0['query']}\"")
    print("=" * 75)
    res0 = run_pipeline(q0["query"], q0["pool"], qid=q0["qid"], mode="predictive")
    for step in res0["history"]:
        print(f"  --> {step}")
    top0 = res0["top_item"]
    print(f"Outcome: {res0['verifier_feedback']}")
    print(f"Recommended: [{res0['ranked_item_ids'][0]}] {top0['title']} | ${top0['price']} | {top0['ram_gb']}GB")

    # Test 2: Query 3 (Demonstrating Cascade Fallback when E1 fails)
    q3 = qs[3]
    print("\n" + "=" * 75)
    print("TEST 2: CASCADE ROUTER ON QUERY 3 (Demonstrating Self-Correction / Fallback)")
    print(f"Query: \"{q3['query']}\"")
    print("=" * 75)
    res3 = run_pipeline(q3["query"], q3["pool"], qid=q3["qid"], mode="cascade")
    for step in res3["history"]:
        print(f"  --> {step}")
    top3 = res3["top_item"]
    print(f"Outcome: {res3['verifier_feedback']}")
    print(f"Recommended: [{res3['ranked_item_ids'][0]}] {top3['title']} | ${top3['price']} | {top3['ram_gb']}GB")

    # Export Mermaid diagram for slides/thesis
    graph = build_mixragrec_graph()
    try:
        mermaid_src = graph.get_graph().draw_mermaid()
        with open("data/graph_architecture.mmd", "w", encoding="utf-8") as f:
            f.write(mermaid_src)
        print("\n" + "=" * 75)
        print("Exported LangGraph architecture diagram -> data/graph_architecture.mmd")
        print("=" * 75)
    except Exception as e:
        print(f"Mermaid export notice: {e}")
