# ADR-2026-05-21: First Advanced TO-BE Track — Self-Calibrating Cost Model

- Date: 2026-05-21
- Status: Proposed
- Decision owners: MLCouncil quant/platform
- Related PR/Issue: Prompt 10 (advanced track selection); architecture `docs/architecture-as-is-to-be-2026-05-21.md` M9; Prompt 07 Stage B

## Context

After P0 foundation cleanup and a reproducible baseline (`docs/internal/baselines/`), MLCouncil should adopt its first **advanced** track without introducing a disruptive alpha or council model. Five candidates were evaluated against repository evidence (2026-05-21):

| Track | Code evidence | Disruption |
|-------|---------------|------------|
| TFT/PatchTST alpha | Docs/PDF only; `models/technical.py` is LightGBM | High |
| MoE council gating | Docs/PDF only; `council/aggregator.py` uses regime YAML + EWM IC-Sharpe | High |
| HRP / robust portfolio | Docs/PDF only; `council/portfolio.py` is CVXPY mean-variance | Medium |
| Self-calibrating cost | Heuristic `council/transaction_costs.py`; OMS emits `slippage_bps` / `execution_cost_bps` | Low |
| Dashboard math-trace | Attribution UI exists; no formula/constraint waterfall | Low (UI) |

Mismatch **M9** (High): docs overstate Almgren-Chriss while runtime uses static bps heuristics. Paper fills are already captured in `execution/oms.py` but are not fed back into `TransactionCostModel` or backtest defaults. Champion/challenger promotion gates (`council/mlflow_utils.py`, `backtest/validation.py`) exist for **models**, not for cost parameters.

## Decision

Adopt **self-calibrating execution cost model (Stage B implementation)** as the first advanced track. Keep LightGBM, FinBERT, and HMM council paths unchanged. Deliver:

1. A fill-ingestion contract from OMS/paper execution → per-ticker slippage observations.
2. A rolling calibration layer (e.g., EWM or Bayesian shrink toward liquidity priors) updating effective `slippage_bps` used by `TransactionCostModel`, portfolio TC penalty, and backtest defaults.
3. Rollback to static env defaults when sample count or stability checks fail.
4. Promotion evidence: gross vs net Sharpe/turnover delta on walk-forward with calibrated vs static costs (not model promotion gate).

Detailed parameter design remains in a sibling ADR: `docs/adr/YYYY-MM-DD-self-calibrating-cost-model.md` (Prompt 07 Stage B).

## Consequences

**Positive**

- Closes M9 honestly: backtest and optimizer use costs informed by realized paper fills.
- Low blast radius: no new DL dependency, no retrain of alpha models.
- Reuses existing telemetry and `tests/test_backtest.py::TestTransactionCostRealism`.

**Trade-offs**

- Early paper volume may be too sparse for stable per-ticker calibration; global or bucket-level kappa may be required initially.
- Calibrated costs can reduce apparent backtest Sharpe (truthful net), which is desirable before alpha challengers.

**Operational**

- Log calibration version in order/execution lineage alongside existing `LINEAGE_COLUMNS`.
- Alert when calibrated slippage diverges >N bps from static prior for 5+ sessions.

## Alternatives Considered

1. **TFT/PatchTST alpha** — Highest IC upside in roadmap (+0.02–0.05 IC aspirational) but requires torch, sequence datasets, months of validation; violates “no disruptive model” for first track.
2. **MoE council gating** — Touches core decision layer; REINFORCE/PPO gating is research-heavy with correlated experts risk.
3. **HRP / robust portfolio** — Good drawdown story; still changes optimizer behavior without fixing cost honesty; defer until cost-aligned baseline exists.
4. **Dashboard math-trace** — Critical for operator trust (P1 observability); best parallel enabler, not first quant track. Depends on stable diagnostics contracts post-baseline.

## Rollout Plan

1. **Gate**: P0 doc/config reconciliation complete; publish `docs/internal/baselines/YYYY-MM-DD-clean-baseline.md`.
2. **Design**: Author `docs/adr/YYYY-MM-DD-self-calibrating-cost-model.md` (min fills, update rule, rollback).
3. **Implement**: Ingest OMS fills → calibration store → wire into `TransactionCostModel.from_env()` / calibrated override.
4. **Validate**: Walk-forward gross/net comparison; paper trading reconciliation report.
5. **Next track** (after baseline + math-trace MVP): HRP soft prior or dashboard constraint waterfall.

## Verification

- `python -m pytest tests/test_backtest.py -k TransactionCost -v`
- `python -m pytest tests/test_trading_service.py -v`
- New: `tests/test_cost_calibration.py` (deterministic fill fixtures, rollback, monotonicity)
- Walk-forward artifact: calibrated vs static `oos_sharpe`, turnover, implementation shortfall bps
- No change to `validate_promotion_gate` until a separate cost-parameter policy is defined
