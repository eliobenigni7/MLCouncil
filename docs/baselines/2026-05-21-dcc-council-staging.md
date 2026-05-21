# Baseline — DCC-GARCH covariance (T3.4 staging)

Date: 2026-05-21

## Promotion

```bash
python scripts/promote_council_module.py --module dcc
```

Sets `council.covariance_estimator: dcc` in `config/production_manifest.yaml`.

## Rollback

```bash
python scripts/promote_council_module.py --module ledoit  # or hand-edit manifest to ledoit
```

Note: use manifest edit for ledoit — script supports `moe`, `cqr`, `dcc`, `diff`, `stacking` only.
