# Baseline — Walk-forward foundation (T1.1)

Date: 2026-05-21  
Profile: staging / CI seed

## Scope

Champion/challenger gate for `lightgbm`, `sentiment`, `hmm`, `tft` via
`scripts/run_walkforward_promotion.py` and Dagster `model_promotion_gate`.

## Verification

```bash
python scripts/populate_walkforward_caches.py
python scripts/run_walkforward_promotion.py --model lightgbm --json
python -m pytest tests/test_walkforward_promotion.py -v
```

## Rollback

Disable `.github/workflows/walk-forward-ci.yml` or set all matrix jobs to `--dry-run`.
