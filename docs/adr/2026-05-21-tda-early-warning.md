# ADR: TDA Early Warning (T4.5 Shadow)

- Date: 2026-05-21
- Status: Accepted (scaffolding)

## Decision

`council/tda_warning.py` + Dagster `tda_warning_signal` (weekly Monday 06:00 UTC).
Uses correlation-loop proxy when `gudhi`/`ripser` not installed.

## Rollback

`MLCOUNCIL_TDA_WARNING_ENABLED=false` or disable `tda_warning_job` schedule.
