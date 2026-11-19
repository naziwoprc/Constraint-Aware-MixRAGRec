# Constraint-Aware MixRAGRec for Laptop Recommendation

> Master's course project — Multi-Agent RAG for Product Recommendation

## Research Question

Can extracting explicit user constraints (hard vs. soft) determine how much
product knowledge to retrieve, achieving better constraint satisfaction at
lower cost?

## Based On

**MixRAGRec** — *Mixture-of-Experts Knowledge Graph Retrieval-Augmented
Generation for Multi-Agent LLM-based Recommendation* (KDD 2026)  
Original repo: [github.com/Sjay-Wang/MixRAGRec](https://github.com/Sjay-Wang/MixRAGRec)

## Key Idea

Instead of always retrieving full product specs (expensive) or nothing (cheap
but inaccurate), a **router** selects the cheapest retrieval level that
satisfies the query:

| Expert | What the LLM sees | Avg tokens |
|--------|-------------------|------------|
| **E1** | Title only | ~1,300 |
| **E2** | Title + query-relevant specs | ~1,800 |
| **E3** | Title + full spec card | ~2,450 |

## Final Benchmark Results (198 Test Queries)

| Model / Baseline | Top-1 Accuracy | Avg Prompt Tokens | Token Savings vs E3 | E1 Routing Rate |
| :--- | :---: | :---: | :---: | :---: |
| **Fixed E1** (Title Only) | 85.4% | 1,283 | +47.1% | 100.0% |
| **Fixed E2** (Targeted Specs) | 99.5% | 1,776 | +26.8% | 0.0% |
| **Fixed E3** (Standard Full RAG) | 100.0% | 2,424 | 0.0% (ref) | 0.0% |
| **Random Router** (50/50) | 92.4% | 1,529 | +36.9% | 50.0% |
| **Trained Router** (Ours - Single-turn) | **98.0%** | **1,597** | **+34.1%** | 35.9% |
| **Cascade + Verifier** (Ours - Guardrail) | **99.5%** | **1,544** | **+36.3%** | 85.4% |
| **Oracle Router** (Theoretical Ceiling) | 100.0% | 1,357 | +44.0% | 85.4% |

**Key Takeaways:**
1. **Fixed E3 (Standard RAG)** is highly redundant, wasting ~1,140 extra tokens per query compared to E1.
2. **Trained Router** achieves **98.0% accuracy** while saving **34.1% of prompt tokens** in a single round of generation.
3. **Cascade with Verifier** achieves near-perfect **99.5% accuracy** with **36.3% token savings**, providing deterministic constraint guarantees.

## Pilot Results (100 train queries)

| Method | Top-1 Accuracy | Avg Prompt Tokens |
|--------|---------------|-------------------|
| E1 (title only) | 79% | 1,307 |
| E2 (relevant specs) | 100% | 1,789 |
| E3 (full specs) | 100% | 2,451 |
| Oracle Router | 100% | 1,403 |

The oracle router picks E1 for 79% of queries and E2 for the remaining 21%,
achieving **100% accuracy at 43% fewer tokens** than E3.

## Architecture

```
User Query
    │
    ▼
┌──────────────────┐
│ Constraint        │  LLM extracts hard constraints
│ Extractor         │  (price, RAM, storage, brand)
│ (extractor.py)    │  and soft preferences
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Router            │  Selects cheapest expert
│ (train_router.py) │  likely to succeed
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Expert E1/E2/E3   │  Renders candidate info
│ (experts.py)      │  at selected granularity
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ LLM Ranker        │  Ranks top-5 from 20
│ (pilot.py)        │  candidates
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Hard Constraint    │  Deterministic check:
│ Verifier           │  does top-1 satisfy
│ (deterministic)    │  all hard constraints?
└──────────────────┘
```

## Dataset

- **Source:** Amazon Reviews 2023 (McAuley Lab), Electronics → Laptops
- **Items:** 1,042 cleaned laptops with price, RAM, storage, weight, screen, brand
- **Queries:** 600 synthetic queries with ground-truth constraints
  - Pool: 20 candidates each (5 valid, 10 near-miss, 5 far)
  - Random baseline: ~25% top-1
- **Paraphrases:** 145 rewritten queries in 6 styles (chatty, keyword, formal, etc.)
- **Split:** train=303, dev=99, test=198 (grouped by constraint set, seed=42)

## Project Structure

```
code/
├── .env                  # API keys (not committed)
├── .gitignore
├── README.md             # This file
│
├── data/
│   ├── laptops_clean.parquet    # 1,042 cleaned laptops
│   ├── cards.json               # Structured spec cards
│   ├── queries.jsonl            # 600 queries with ground truth
│   ├── queries_para.jsonl       # 150 paraphrased queries
│   ├── queries_para_clean.jsonl # 145 validated paraphrases
│   ├── split.json               # train/dev/test split (DO NOT regenerate)
│   └── pilot_results.json       # Pilot: 100 train × 3 experts
│
├── check_data.py         # Data exploration & cleaning → laptops_clean.parquet
├── build_cards.py        # Build spec cards → cards.json
├── make_split.py         # Create train/dev/test split → split.json
├── paraphrase.py         # LLM paraphrasing → queries_para.jsonl
├── clean_para.py         # Filter bad paraphrases → queries_para_clean.jsonl
│
├── extractor.py          # Constraint Extractor (LLM + Pydantic + cache)
├── eval_extractor.py     # Evaluate extractor on template queries
├── eval_para.py          # Evaluate extractor on paraphrased queries
│
├── experts.py            # E1/E2/E3 rendering logic
├── pilot.py              # Run single expert on queries
├── run_all.py            # Run all experts, compute oracle router stats
├── test.py               # API provider speed test
│
├── collect.py            # [TODO] Collect all results → router_data.jsonl
├── train_router.py       # [TODO] Train router model
└── evaluate.py           # [TODO] Full evaluation with all baselines
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install openai pydantic diskcache tenacity python-dotenv pandas pyarrow
```

Create `.env`:
```
NANOGPT_API_KEY=sk-nano-...
NANOGPT_PROVIDER=deepinfra
```

## LLM

- **Model:** `deepseek/deepseek-v4.1-flash-1017`
- **API:** NanoGPT (OpenAI-compatible, `https://nano-gpt.com/api/v1`)
- **Provider:** `coralbricks-standard` via `X-Provider` header

## Simplifications vs. Original Paper

| Aspect | MixRAGRec (original) | This project |
|--------|---------------------|--------------|
| Knowledge structure | Knowledge Graph (triples, subgraphs) | Tabular spec cards |
| LLM | LLaMA-3-8B / Mistral-7B (local) | DeepSeek V4.1 Flash (API) |
| Router training | PPO/GAE with MMAPO reward | Cost-Sensitive Calibrated Router |
| Contrastive learning | DPO-style fine-tuning | Not implemented |
| Alignment agent | LLM rewriting | Deterministic formatting |
| Candidate pool | 1 target + 19 distractors | 5 valid + 10 near-miss + 5 far |
| Queries | From user history | Synthetic with ground truth |
| Hardware | 96 GB GPU | RTX 3060 6GB + API |

## Status

- [x] Data collection & cleaning
- [x] Query generation (600 template + 145 paraphrased)
- [x] Constraint Extractor (97% accuracy)
- [x] Three expert levels (E1/E2/E3)
- [x] Pilot experiment (100 train queries)
- [x] Full data collection (all splits x all experts)
- [x] Router training & tuning
- [x] Cascade baseline
- [x] Full evaluation & comparison
- [ ] Multi-Agent LangGraph Pipeline (Demo)
- [ ] Final report

## License

Course project — not for redistribution.
