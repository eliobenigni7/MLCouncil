# ADR-0007: FinMA/FinGPT + RAG Sentiment Challenger (Shadow Mode)

- Date: 2026-05-21
- Status: Accepted (scaffolding)
- Decision owners: MLCouncil quant platform
- Related: Wave 2 track T2.2 (`docs/internal/disruptive-roadmap-2026-05-21.md`)

## Context

Daily production sentiment uses FinBERT (`models/sentiment.py`) in
`sentiment_features` → `sentiment_signals`. FinBERT is fast and stable but
limited to headline text and a 2019-era finance vocabulary. T2.2 explores a
finance-tuned LLM (FinMA-7B / FinGPT) with retrieval over SEC 10-K/10-Q
passages to improve IC and event lift without destabilizing the live council path.

Champion promotion remains walk-forward gated (`council/walkforward_promotion_gate.py`);
the LLM stack runs in **shadow mode** until IC delta ≥ +0.02 and throughput SLOs pass.

## Decision

1. **`models/sentiment_llm.py`** — `LLMSentimentScorer`:
   - RAG prompt via `VectorStore.retrieve()` (top-K passages).
   - Backends: injected fn (tests), `llama-cpp` when `MLCOUNCIL_LLM_GGUF_PATH` set,
     else keyword mock (no GPU).
   - **Hallucination guard:** `parse_sentiment_score()` failure → `0.0`.
   - `log_shadow_scores()` writes `data/results/shadow_sentiment_llm/{date}.parquet`.
2. **`data/ingest/sec_filings.py`** — EDGAR REST skeleton (`data.sec.gov` JSON);
   complements existing `data/ingest/edgar_ingest.py` (class-based cache) with a
   thin functional API for RAG indexing.
3. **`data/store/vector_store.py`** — Chroma persistent client when installed;
   in-memory mock otherwise (`MLCOUNCIL_VECTOR_STORE_MOCK=true` forces mock).
4. **Production path unchanged** — FinBERT remains authoritative for
   `sentiment_signals`; enable shadow with `MLCOUNCIL_LLM_SENTIMENT_SHADOW=true`
   in scripts or a future Dagster hook (not wired to council weights in v1).

## Gating (promotion, not yet implemented)

| Criterion | Threshold |
|-----------|-----------|
| IC delta vs FinBERT | ≥ +0.02 on 1d forward returns |
| Event lift | ≥ +20 bps avg return post strong positive sentiment days |
| Throughput | ≥ 100 headlines/s (Q4_0 quantized) |
| Hallucination guard | Unparseable LLM output → score 0 |

## Consequences

- Positive: offline comparison of LLM+RAG vs FinBERT without council risk.
- Trade-off: optional deps (`chromadb`, `llama-cpp-python`) not required for CI;
  mock backends keep pytest fast.
- Operations: shadow logs under `data/results/shadow_sentiment_llm/` feed
  walk-forward signal caches when populated.

## Alternatives Considered

1. **Replace FinBERT in `sentiment_signals` immediately** — rejected (no gating).
2. **Qdrant instead of Chroma** — deferred; Chroma optional dep matches roadmap.
3. **Whisper earnings transcripts (T2.2 file list)** — deferred to follow-up PR.

## Rollout Plan

1. Land scaffolding + tests on branch `feat/finma-rag-sentiment`.
2. Index SEC passages per ticker into `VectorStore` (batch job, not daily SLA).
3. Enable shadow logging in staging; populate IC comparison notebooks.
4. Promote only via walk-forward gate after criteria pass.

## Verification

```bash
python -m pytest tests/test_sentiment_llm.py tests/test_sec_filings.py -v
# optional local LLM:
# export MLCOUNCIL_LLM_GGUF_PATH=/path/to/finma-q4.gguf
# export MLCOUNCIL_LLM_SENTIMENT_SHADOW=true
```

## Rollback

- Unset `MLCOUNCIL_LLM_SENTIMENT_SHADOW` (default off).
- Delete shadow parquet under `data/results/shadow_sentiment_llm/` if needed.
- Remove optional `chromadb` / `llama-cpp-python` from environment without affecting FinBERT.
