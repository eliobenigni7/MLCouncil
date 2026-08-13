# ADR-2026-08-13: Unified Risk Alerting (Immune System)

- Date: 2026-08-13
- Status: Accepted
- Decision owners: MLCouncil core
- Related: roadmap F-0.2, `docs/math-drilldown-2026-2030-autonomous-council.md` §2,
  ADR `2026-05-21-causal-drift-pcmci.md`, ADR `2026-05-21-tda-early-warning.md`

## Context

The system had four independent drift/warning families — TDA early warning
(`council/tda_warning.py`, enabled by default), causal graph drift
(`council/causal_drift.py`, `check_causal_graph_drift` in `council/monitor.py`),
streaming ADWIN/DDM (`council/drift.py`), and evidently dataset drift
(`council/evidently_reports.py`) — but no operator surface that combined them.
In particular:

- `check_causal_graph_drift` was exported from `council/__init__.py` but wired
  nowhere: no Dagster asset, no schedule, no API endpoint. Its weekly cadence
  never ran.
- The causal drift detector had a latent bug for scheduled use: a fresh detector
  was created every run and never persisted a baseline, so the change-fraction
  metric could never alert across runs.
- The fragile pattern in `data/pipeline.py` mutating `os.environ`
  (`MLCOUNCIL_AGGREGATOR_MODE` switch-and-restore around council aggregation)
  was a global-state hazard.
- `council/monitor.py::_to_pandas` silently returned empty DataFrames for
  `pd.Series` inputs, breaking `check_causal_graph_drift` on its documented
  input type.

## Decision

1. **Weekly causal drift check in Dagster.** New asset `causal_drift_check`
   mirroring the `tda_warning_signal` pattern (output
   `data/results/causal_drift_latest.json`), plus job and schedule
   `causal_drift_schedule` (Mondays 02:00 UTC, aligned with walk-forward CI).
   Baseline persistence added: `PCMCIDriftDetector.set_baseline` /
   `save_causal_baseline` / `load_causal_baseline`
   (`data/results/causal_baseline.json`) so the change fraction is computed
   against a stable reference across runs. `failure_sensor` extended to cover
   `[daily_job, tda_warning_job, causal_drift_job]`.
2. **Unified health aggregation.** New `council/alerting.py`:
   `collect_health_signals()` merges the five signal families into
   `{signal: {level: ok|warn|alert, value, threshold, note}}` with graceful
   degradation (missing inputs → `ok` + note, never raises).
   Thresholds: causal change fraction 0.25, evidently drift fraction 0.5,
   TDA β₁-proxy 0.35.
3. **Dispatch through the existing alert pipeline.** Alert-level signals are
   routed via `council/alerts.py` `AlertDispatcher` (log files, dashboard
   state, CRITICAL email), on the weekly asset cadence — the GET endpoint stays
   read-only. This composes with, and does not duplicate, the existing
   `monitor.py` checks that already emit `AlertResult`.
4. **Read-only API surface.** `GET /api/monitoring/health` returns the
   aggregated signals from `data/results/*.json` via
   `monitoring_service.get_health_signals()`. No side effects on GET.
5. **Clean injection for aggregator mode.** `CouncilAggregator.aggregate(..., aggregator_mode_override=None)`
   (keyword-only, backward compatible; env remains the fallback). The pipeline
   passes the mode explicitly; the `os.environ` mutation is removed. No
   subprocess reads that env in the affected path.
6. **Bug fix in monitor.** `_to_pandas` handles `pd.Series` inputs correctly.

## Consequences

- Positive: single operator surface for regime/drift health; causal check now
  runs weekly with a persistent baseline; alerts reach the same channels
  (logs, dashboard state, email) as all other council alerts; no global env
  mutation.
- Trade-offs: causal drift math remains the simplified Pearson-threshold
  version (upgrade to partial-correlation PC skeleton is P-3.4 / math
  drill-down §2.4); baseline is a JSON snapshot — first run establishes it,
  so no alert can fire on run #1.
- Operational: dashboard shows health signals; runbook should treat a
  `causal_drift` alert as "graph structure changed" and review the link set
  before acting; email dispatch cadence is weekly.

## Alternatives Considered

1. Wire checks through `council/monitor.py` only (no new module) — rejected:
   `monitor.py` is already 1000+ lines and mixes MLflow/alert side effects with
   pure aggregation.
2. ContextVar for aggregator mode — rejected: implicit cross-module state;
   keyword-only parameter is explicit and testable.
3. Dispatch from the API endpoint — rejected: GET must be side-effect-free.

## Rollout Plan

1. Ship with Phase 0 (done: asset, schedule, endpoint, tests).
2. Observe two weekly runs in staging; verify baseline persistence and alert
   behavior by injecting a synthetic link change.
3. Surface health signals on the dashboard in a later dashboard workstream.

## Verification

- `tests/test_monitor.py` (44 tests, incl. 19 new: causal drift wiring,
  health signals), `tests/test_pipeline.py` (73, incl. causal asset),
  `tests/test_api_monitoring.py` (8, incl. `/health` with/without artifacts) —
  all green; 161-test combined Phase 0 run green.
- Known pre-existing failures unrelated to this change:
  `tests/test_drift.py::TestADWINDetector::test_update_reflects_detector_drift_flag`
  and `TestDDMDetector::test_ddm_reads_binary_drift_flag` (stale tests
  referencing a nonexistent `_detector` attribute; `council/drift.py` untouched).
