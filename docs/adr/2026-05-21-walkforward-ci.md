# ADR-0004: Walk-Forward Champion/Challenger CI for Alpha Models

- Date: 2026-05-21
- Status: Accepted
- Decision owners: MLCouncil quant platform
- Related PR/Issue: Wave 1 track T1.1 (`docs/internal/disruptive-roadmap-2026-05-21.md`)

## Context

Disruptive alpha challengers (TFT, FinMA, deep regime, microstructure) cannot be
promoted safely without a repeatable, purged walk-forward re-fit and explicit
champion/challenger gating. Cost calibration already uses
`validate_cost_calibration_promotion`; alpha models need the same credibility
layer before any checkpoint replaces production inference.

The daily Dagster pipeline must keep serving the **champion** only. Challengers
run in **shadow mode** (train, evaluate, log) until promotion criteria pass.

## Decision

Introduce a **weekly GitHub Actions workflow** and a local orchestrator that:

1. Retrain a shadow challenger checkpoint (per model family).
2. Evaluate purged+embargoed walk-forward metrics via `run_walk_forward_analysis`.
3. Gate promotion with `validate_model_promotion` using:
   - `oos_sharpe >= champion - 0.1`
   - `pbo <= 0.5`
   - `walk_forward_window_count >= 8`
4. Persist reports under `data/operations/walkforward_promotion_{model}.json`.
5. Track **three consecutive passing runs** before auto-promote eligibility
   (manual PR still required; no automatic pipeline wiring).

Cadence:

- **LightGBM**: weekly (Monday 02:00 UTC) via `.github/workflows/walk-forward-ci.yml`
- **FinBERT sentiment**: monthly retrain recommended; weekly CI uses dry-run +
  cached shadow signals until a dedicated monthly workflow is added
- **HMM regime**: same as sentiment until `train_regime` script lands

Champion archive: promoted SHA recorded in gate report / git tag at promote time
(operator step; not automated in v1).

## Consequences

- Positive: all future T2.x challengers reuse one promotion path; lookahead-safe
  splits are mandatory before merge.
- Trade-off: CI runs dry-run without full GPU retrain; production weekly job
  needs cached `data/results/walkforward_signals_*.parquet` populated by shadow
  logging or backtest refresh.
- Operations: failed gates do not change daily pipeline; streak counter resets
  on failure.

## Alternatives Considered

1. **Manual quarterly retrain** — low automation, no audit trail (rejected).
2. **Promote on single OOS Sharpe bump** — overfits noise; PBO + window count
   required (rejected).
3. **Wire challengers into council immediately** — violates no-big-bang policy
   (rejected).

## Rollout Plan

1. Land workflow + `scripts/run_walkforward_promotion.py` + validation gate.
2. Populate champion metric JSON + signal caches from
   `scripts/run_strategy_backtest.py` refresh.
3. Enable non-dry-run retrain on self-hosted runner when OHLCV + GPU available.
4. Open "promote model" PR only when streak ≥ 3 and gate passes.

## Verification

```bash
gh workflow run walk-forward-ci.yml --ref master
python scripts/run_walkforward_promotion.py --model lightgbm --dry-run
python -m pytest tests/test_walkforward_promotion.py -v
```

## Rollback

- Disable workflow in GitHub UI or delete `.github/workflows/walk-forward-ci.yml`.
- Daily pipeline continues using `models/checkpoints/*_latest.pkl` champions.
- Remove `data/operations/walkforward_streak_*.json` to reset promotion streak.
