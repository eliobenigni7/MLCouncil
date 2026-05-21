# ADR: Production Setup with Walk-Forward Promotion Gate

- Date: 2026-05-21
- Status: Accepted
- Related: T1.1 walk-forward CI, `config/production_manifest.yaml`

## Context

Frontier profile enables all disruptive flags immediately, bypassing champion/challenger
discipline. Production paper trading must:

1. Run **champions only** by default (LightGBM, FinBERT, HMM).
2. Evaluate challengers in **shadow** via weekly walk-forward gate.
3. Promote to production only after **3 consecutive gate passes** + `promote_model.py`.

## Decision

### Production profile

- `MLCOUNCIL_ENV_PROFILE=prod` loads `config/runtime.prod.env`.
- `MLCOUNCIL_USE_PRODUCTION_MANIFEST=true` — council/portfolio flags come from
  `config/production_manifest.yaml`, not ad-hoc env toggles.

### Promotion flow

```
Weekly CI / Dagster model_promotion_gate
  → run_walkforward_promotion.py logic
  → data/operations/walkforward_promotion_{model}.json
  → streak counter (3 passes)

Operator
  → python scripts/promote_model.py --model lightgbm
  → copy challenger → champion checkpoint
  → update production_manifest.yaml + promotion_history
```

### Frontier vs prod

| Profile | Purpose |
|---------|---------|
| `prod` / `paper` | Gated champions, manifest-driven |
| `frontier` | R&D — all flags on, no promotion required |

## Rollback

- Revert `production_manifest.yaml` from git.
- Restore previous `models/checkpoints/*_latest.pkl` from backup.
- Set `MLCOUNCIL_AUTOMATION_PAUSED=true`.

## Verification

```bash
python scripts/setup_prod.py
python scripts/run_walkforward_promotion.py --model lightgbm --dry-run
python scripts/promote_model.py --model lightgbm  # after 3 passes
```
