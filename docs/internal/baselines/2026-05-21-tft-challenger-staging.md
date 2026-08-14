# Baseline — TFT alpha challenger (T2.1 staging)

Date: 2026-05-21  
Status: staging promotion path (not empirical GPU walk-forward)

## Commands

```bash
python scripts/populate_walkforward_caches.py --models tft,lightgbm
python scripts/establish_wave2_staging_promotion.py --model tft
python scripts/promote_model.py --model tft --force  # after reviewing gate JSON
```

## Production manifest

After promotion: `models.technical.family=tft`, `experts.tft.enabled=true`.

## Rollback

```bash
# Revert manifest models.technical to lightgbm
export MLCOUNCIL_USE_PRODUCTION_MANIFEST=true
# experts.tft.enabled=false
```
