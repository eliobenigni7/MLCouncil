# ADR: Stacking Meta-Learner + CQR Position Sizing (T3.2 Shadow)

- Date: 2026-05-21
- Status: Accepted (shadow scaffold)
- Related: `docs/disruptive-roadmap-2026-05-21.md` Wave 3 T3.2

## Context

MAPIE Jackknife+ (`council/conformal.py`) gives marginal coverage but wide
intervals in volatile regimes. CQR targets conditional coverage; a stacking
meta-learner can combine 3+ base model outputs before sizing.

## Decision

1. **`council/cqr.py`** — `CQRPositionSizer`, `StackingMetaLearner` (Ridge default;
   optional XGB via `MLCOUNCIL_STACKING_BACKEND=xgb`).
2. **`MLCOUNCIL_POSITION_SIZING`** — `conformal` (default) or `cqr`.
3. **`get_position_sizer()`** factory; pipeline loads `conformal_sizer.pkl` or `cqr_sizer.pkl`.
4. **`scripts/train_stacking_cqr.py`** fits checkpoints; **`MLCOUNCIL_STACKING_SHADOW=true`**
   logs `data/results/shadow_stacking/{partition}.parquet` from `council_signal`.

## Gating (promotion)

- Conditional coverage 80–90% per volatility quintile.
- Mean interval width ≤ Jackknife+ width × 1.05.

## Rollback

`MLCOUNCIL_POSITION_SIZING=conformal` (default).

## Verification

```bash
python scripts/train_stacking_cqr.py --cqr --stacking
python -m pytest tests/test_cqr.py -v
```
