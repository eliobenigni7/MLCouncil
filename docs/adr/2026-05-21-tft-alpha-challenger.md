# ADR-0008: TFT Alpha Challenger (Shadow Mode)

- Date: 2026-05-21
- Status: Accepted
- Decision owners: MLCouncil quant platform
- Related: Wave 2 track T2.1 (`docs/internal/disruptive-roadmap-2026-05-21.md`)

## Context

LightGBM (Alpha158 + CPCV) is the production technical alpha. Wave 2 T2.1
requires a **Temporal Fusion Transformer** challenger to capture cross-time
structure without modifying the daily Dagster council path until walk-forward
promotion (T1.1) passes.

Full `pytorch-forecasting` + Lightning adds heavy transitive deps and long CI
install times. A **PyTorch-native TFT-inspired** module (VSN + GRU + attention +
pinball quantiles) provides train/infer parity with the roadmap API while
keeping CPU tests fast.

## Decision

1. **`models/tft.py`** — `TemporalFusionAlpha` (`BaseModel` contract):
   fit/predict, variable-selection weights, `write_shadow_signals()`,
   `build_shadow_signal_matrix()`, `measure_inference_latency_ms()`.
2. **`scripts/train_tft.py`** — offline trainer; writes
   `models/checkpoints/tft_challenger.pkl`,
   `data/results/tft_shadow_signals.parquet`, and
   `data/results/walkforward_signals_tft.parquet` for CI gate.
3. **`config/models.yaml`** — `tft:` hyperparameters (encoder length, quantiles).
4. **Shadow only** — no changes to `data/pipeline.py::lgbm_signals` or
   `council/aggregator.py`. Champion remains LightGBM.
5. **Promotion** — `council/walkforward_promotion_gate.py` adds model key `tft`
   (champion metrics = LightGBM; challenger signals from TFT cache).
6. **Dependency** — `torch>=2.0` in `requirements.txt` (no `pytorch-forecasting`
   in v1; revisit if full TFT library needed for production quantiles).

## Inference SLO

Documented target: **<300 ms CPU** for a daily-sized batch after fit.
Measured on the unit-test fixture (6 tickers, short encoder) via
`measure_inference_latency_ms()`; production batch (~20 tickers × encoder 20)
should be profiled after first full `train_tft.py` run.

## Gating (T1.1 + roadmap T2.1)

Promotion thresholds remain in walk-forward CI (not auto-wired):

- `oos_sharpe_tft >= oos_sharpe_lgbm + 0.15` (12m walk-forward)
- `oos_max_drawdown_tft <= oos_max_drawdown_lgbm + 2%`
- VSN top-10 Jaccard ≥ 0.6 (future monitor hook)
- Inference <300 ms CPU daily batch

Until gate passes: **shadow_mode** only.

## Consequences

- Positive: reusable challenger path for T2.x; CPU CI coverage; clear parquet
  contract for walk-forward signals.
- Trade-off: simplified architecture vs full Lim et al. TFT; quantile head is
  linear on attention context (not full static covariate encoders).
- Operations: weekly `python scripts/run_walkforward_promotion.py --model tft`
  after populating `walkforward_forward_returns.parquet`.

## Alternatives Considered

1. **`pytorch-forecasting.TemporalFusionTransformer`** — rejected for v1 due to
   install weight and TimeSeriesDataSet coupling to full panel schema.
2. **PatchTST** — deferred; TFT track name retained for roadmap alignment.
3. **Immediate council wiring** — rejected (no-big-bang policy).

## Rollout Plan

1. Land scaffold + tests on `feat/tft-alpha`.
2. Run `python scripts/train_tft.py` on staging OHLCV; inspect shadow parquet.
3. Populate forward returns cache; run walk-forward promotion dry-run.
4. If gating passes for 3 consecutive CI weeks → separate PR to promote champion.

## Verification

```bash
pip install -r requirements.txt
python -m pytest tests/test_tft.py -v
python scripts/train_tft.py --start 2021-01-01 --max-epochs 5
python scripts/run_walkforward_promotion.py --model tft --dry-run
```

## Rollback

- Do not call `train_tft.py` in Dagster schedules.
- Delete `data/results/tft_shadow_signals.parquet` and TFT checkpoints.
- Remove `tft` from `SUPPORTED_MODELS` if abandoning track (ADR → Rejected).
