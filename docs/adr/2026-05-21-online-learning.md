# ADR-0005: Online Learning Scaffolding for LightGBM Daily Inference

- Date: 2026-05-21
- Status: Accepted
- Decision owners: MLCouncil quant platform
- Related PR/Issue: Wave 1 track T1.2 (`docs/disruptive-roadmap-2026-05-21.md`)

## Context

The daily Dagster path loads a static champion checkpoint (`lgbm_latest.pkl`)
trained offline with CPCV. Regime shocks require faster adaptation than weekly
walk-forward promotion alone. We need incremental daily updates with automatic
rollback when quality degrades, without bypassing champion/challenger gating.

Walk-forward CI (T1.1) remains the **only** path to promote a new champion
checkpoint from shadow retrain. Online learning adjusts the **existing**
champion in place on recent labeled features only.

## Decision

1. **`models/online.py`** — `IncrementalLightGBM` wraps `TechnicalModel` and
   calls `Booster.refit()` on a rolling labeled window (default 60 calendar days
   of history, 10-day IC holdout).
2. **`council/drift.py`** — `ADWINDetector` and `DDMDetector` (River, CC0) on
   equal-weight portfolio daily returns; ADWIN uses a 60-day rolling feed.
3. **`data/pipeline.py::lgbm_signals`** — when `MLCOUNCIL_ONLINE_LEARNING=true`,
   after loading champion: refit → IC gate → save `lgbm_latest.pkl` + `.hash` on
   pass, else reload champion from disk unchanged.
4. **IC gate** — reject incremental update when
   `IC_today < IC_baseline - 0.05` (configurable via `MLCOUNCIL_ONLINE_IC_THRESHOLD`).
5. **Pickle security** — reuse `TechnicalModel.save()` / `load()` hash sidecars
   from `council/pickle_security.py`.

## Consequences

- Positive: daily adaptation without waiting for weekly CI; ADWIN flags when
  heavy walk-forward retrain should be prioritized.
- Trade-off: incremental refit can diverge on thin or noisy days; IC gate +
  champion reload limits damage.
- Operations: disabled by default; enable per environment with env flag.

## Alternatives Considered

1. **Replace champion inline when ADWIN fires** — bypasses walk-forward gate
   (rejected).
2. **Full CPCV retrain daily** — too expensive for Dagster partition SLA
   (rejected for v1; ADWIN only logs/schedules heavy retrain).
3. **River-only streaming model** — new training stack; defer to T2.x
   (rejected for scaffolding).

## Rollout Plan

1. Land `models/online.py`, `council/drift.py`, pipeline hook, tests, `river` dep.
2. Enable in paper/staging: `MLCOUNCIL_ONLINE_LEARNING=true`.
3. Monitor Dagster metadata `online_learning` on `lgbm_signals` and ADWIN warnings.
4. Keep walk-forward CI as promotion authority.

## Verification

```bash
pip install -r requirements.txt
python -m pytest tests/test_online.py tests/test_drift.py -v
python scripts/run_pipeline.py --partition 2026-05-20 --online
```

## Rollback

- Set `MLCOUNCIL_ONLINE_LEARNING=false` (default) — pipeline reverts to static
  champion load + predict only.
- Restore prior `models/checkpoints/lgbm_latest.pkl` + `.hash` from versioned
  backup if a bad incremental save slipped through (IC gate should prevent this).
- Remove `river` from requirements only after deleting drift detectors.
