# ADR-2026-08-13: Canary Config Profile (F-0.4)

- Date: 2026-08-13
- Status: Accepted
- Decision owners: MLCouncil core
- Related: roadmap F-0.4 / P-1.1, `docs/flag-registry-2026-08-13.md`,
  `docs/math-drilldown-2026-2030-autonomous-council.md` §7.3,
  ADR `2026-08-13-unified-risk-alerting.md` (alert dispatch reuse)

## Context

Phase 0 auditing found ~100 `MLCOUNCIL_*` env flags, of which ~20 disable
shadow features by default. Activation was a binary, ungoverned toggle:
nothing tracked which features were in shadow, which were canary, and when a
feature must be promoted or retired. The roadmap principle P2 (shadow →
canary → production → retirement) and P4 (kill switches everywhere) had no
mechanism. Features like MoE gating existed for months without ever being
activated or evaluated, and nothing would alert if an activated feature
regressed.

## Decision

1. **Canary manifest** `config/canary.yaml`: feature entries
   `{name, env, value, enabled, metrics: {floor, min_days}}`. All features ship
   `enabled: false` — activation is an explicit owner decision at gate G1.
2. **`council/canary.py`** (new):
   - `load_canary_config` — never raises (missing/empty/corrupt → no features);
   - `CanaryState` — JSON persistence at `data/results/canary_state.json`
     (daily metric history, per-feature enabled/revert metadata; corrupt →
     fresh state, never raises);
   - `CanaryController.apply()` — run-policy env injection via `setdefault`
     (operator-set env always wins), called at the root asset of the daily
     job (`raw_ohlcv`) so flags are set before any consumer executes. This is
     a start-of-run policy, distinct from the removed mid-run
     switch-and-restore pattern (F-0.2 ADR). Documented caveat: env injection
     is per-process — Dagster multiprocess configs must apply the manifest in
     each process (future work);
   - `record(date, metrics)` / `check_revert()` — a feature reverts when its
     configured metric is below `floor` for `min_days` consecutive records;
     revert is sticky (persisted) and returns `RevertEvent`;
   - `run_canary_health(...)` — record + check + dispatch revert alerts
     through the F-0.2 `dispatch_health_alerts` pipeline (log + dashboard
     state + CRITICAL email), dispatcher injectable for tests.
3. **Daily asset** `canary_health` (deps: council_signal, portfolio_weights,
   partitioned, retry=2): records same-day council metrics
   (mean|z| of combined signal, portfolio turnover, realized vol proxies);
   no-op with zero side effects when no feature is enabled (no state file
   created, no alerts).
4. **Flag governance**: `docs/flag-registry-2026-08-13.md` inventories all 93
   code flags with default, owner module, purpose, status, target phase and
   expiry date; 22 owner modules carry docstring annotations
   `Canary status: shadow — target: <phase> — expiry: <date> (promote via
   canary o retire)`.

## Consequences

- Positive: activation discipline (a feature exists in the daily path with
  telemetry, or it expires); automatic revert with operator alerting; the
  registry makes the ~100 flags auditable; walk-forward CI remains the
  promotion authority — canary guards are a tripwire, not a validator.
- Trade-offs: revert floors are absolute manifest values, not yet
  baseline-relative (a future refinement could use the pre-enable median);
  env-injection is per-process (multiprocess caveat documented);
  metric proxies are same-day heuristics — the rigorous IC-Sharpe guard
  requires forward returns and is deferred to P-2.3 retirement machinery.
- Operational: G1 decides the activation list; floors are tunable in
  `config/canary.yaml`; a revert produces a CRITICAL alert through the
  standard channels.

## Alternatives Considered

1. Baseline-relative revert (metric < pre-enable median − tolerance) —
   deferred: needs longer history discipline; absolute floors are simpler and
   safe for a tripwire.
2. ContextVar-based feature injection — rejected: env vars are the existing
   contract read by all modules; ContextVar would require touching every
   reader.
3. Full third-party feature-flag system — rejected: no new dependencies;
   scope is a bounded controller, not a platform.
4. Activate canary features by default — rejected: violates gate G1 and P1
   (nothing outside the daily path without evidence).

## Rollout Plan

1. Ship infrastructure (done) with all features disabled.
2. Gate G1: owner approves the activation list; flip `enabled: true` in
   `config/canary.yaml`.
3. Observe daily runs; tune floors per feature; verify revert behavior by
   injecting a synthetic floor breach in staging.
4. Phase 1: add the remaining Wave 3/4 candidates to the manifest.

## Verification

- `tests/test_canary.py` (25 tests: parsing, setdefault precedence, sticky
  revert, no-revert paths, dispatch on revert, no-op without features,
  pipeline asset), `tests/test_pipeline.py` (asset registration, DAG
  acyclicity, partition/retry policies), `tests/test_monitor.py` regression —
  all green; combined Phase 0 run green (270+ tests).
- No-op invariant: `data/results/canary_state.json` absent when no feature
  enabled.
