# ADR: DCC-GARCH Dynamic Covariance (T3.4 Shadow)

- Date: 2026-05-21
- Status: Accepted (shadow scaffold)
- Related: `docs/disruptive-roadmap-2026-05-21.md` Wave 3 T3.4

## Context

Ledoit-Wolf shrinkage on 90-day sample cov is static. DCC-GARCH (Engle 2002)
updates conditional correlation dynamics for portfolio vol targeting.

## Decision

1. **`council/covariance_dynamic.py`** — `DCCEstimator` (GARCH(1,1) per asset via
   `arch`, EWMA correlation recursion).
2. **`data/pipeline._compute_covariance`** — `MLCOUNCIL_COVARIANCE_ESTIMATOR=ledoit`
   (default) or `dcc`.
3. Fallback to sample cov + PSD projection when `arch` missing or panel too short.

## Gating (promotion)

- Realised vol MAPE improves ≥ 10% vs Ledoit-Wolf.
- Max drawdown not worse than baseline by more than 1%.

## Rollback

Default `ledoit` or unset env.

## Verification

```bash
python -m pytest tests/test_dcc_garch.py -v
```
