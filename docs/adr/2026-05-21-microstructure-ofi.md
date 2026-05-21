# ADR-0008: Microstructure / Order Flow Imbalance Alpha (Shadow Challenger)

- Date: 2026-05-21
- Status: Accepted
- Decision owners: MLCouncil quant platform
- Related: Wave 2 track T2.4 (`docs/disruptive-roadmap-2026-05-21.md`)

## Context

The daily council aggregates three alpha families (LightGBM, sentiment, HMM regime
context). TO-BE calls for a fourth intraday signal from L2 order book depth:
**Order Flow Imbalance (OFI)** per Lo & MacKinlay. Live L2 subscriptions (Alpaca
elite, Databento Premium) are not yet budgeted; the track must still land
testable code paths and shadow logging without changing production weights.

Walk-forward CI (T1.1) remains the only promotion path into
`config/regime_weights.yaml` active weights.

## Decision

1. **`intraday/market_data.py`** — `BookLevel`, `BookSnapshot`, `compute_ofi()`
   implementing
   `OFI_t = Δ Σ_b q^b_t − Δ Σ_a q^a_t` on the top 5 bid/ask levels.
2. **`data/ingest/orderbook.py`** — ingest skeleton with `SyntheticOrderBookAdapter`
   (fixture), `AlpacaOrderBookAdapter` and `DatabentoOrderBookAdapter` stubs
   that raise / return deferred until L2 subscription is approved.
3. **`models/microstructure.py`** — `MicrostructureModel` for shadow OFI signals;
   cross-sectional z-score via `predict()`. Not called from `data/pipeline.py`
   council layer in v1.
4. **`config/regime_weights.yaml`** — `microstructure:` block (`enabled: false`,
   `shadow_mode: true`); regime placeholder commented until promotion.
5. **Env flags**
   - `MLCOUNCIL_MICROSTRUCTURE_SHADOW=true` (default) — compute/log only
   - `MLCOUNCIL_MICROSTRUCTURE_PROMOTED=false` (default) — council weights inactive
   - `MLCOUNCIL_ORDERBOOK_FEED=synthetic` (default) | `alpaca` | `databento`

## L2 subscription (deferred)

Live Alpaca/Databento ingest is **explicitly deferred** in code and ADR. Agents
and CI must use `synthetic` feed or `synthetic_book_sequence()` until operations
approves Databento (or equivalent) budget per roadmap risk register.

## Gating (pre-promotion, via T1.1 when wired)

- IC ≥ 0.04 on forward 30-minute returns
- |ρ| ≤ 0.4 vs existing alphas
- Tick-to-signal latency < 500 ms (production SLO)

## Consequences

- Positive: OFI formula and ingest contracts are testable today; shadow mode
  preserves no-big-bang policy.
- Trade-off: no real microstructure edge until L2 feed; synthetic fixture only
  validates arithmetic.
- Operations: zero change to daily Dagster council weights until promotion env
  is set and walk-forward gate passes.

## Alternatives Considered

1. **Wire OFI into aggregator immediately** — violates shadow/champion policy
   (rejected).
2. **Skip ingest module; only unit-test `compute_ofi`** — no path for future
   Alpaca/Databento swap-in (rejected).
3. **Use top-of-book only (1 level)** — diverges from roadmap 5-level spec
   (rejected for v1).

## Rollout Plan

1. Land modules + tests on `feat/microstructure-ofi`.
2. Shadow-log OFI in intraday or offline jobs when `MLCOUNCIL_MICROSTRUCTURE_SHADOW=true`.
3. After L2 subscription: implement Alpaca/Databento adapters behind same interface.
4. Register `microstructure` in walk-forward promotion; on 3× CI pass, set
   `MLCOUNCIL_MICROSTRUCTURE_PROMOTED=true` and enable regime weights.

## Verification

```bash
python -m pytest tests/test_microstructure.py -v
```

Canonical fixture: cumulative bid 300→310, ask 250→255 → **OFI = 5**.

## Rollback

- `MLCOUNCIL_MICROSTRUCTURE_SHADOW=false` — stop shadow compute/logging.
- Leave `MLCOUNCIL_MICROSTRUCTURE_PROMOTED` unset — council ignores model.
- Remove `microstructure` weights from `regime_weights.yaml` if partially enabled.
