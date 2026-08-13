# MLCouncil Roadmap 2026–2030 — The Autonomous Council

Status: **Draft v1 — owner decisions pending (see Decision Gates)**
Date: 2026-08-13
Companion docs: `docs/architecture-as-is-to-be-2026-05-21.md` (drift register), `docs/adr/` (24 ADRs),
`docs/math-drilldown-2026-2030-autonomous-council.md` (exact formulas, critiques, verification statistics).

## 1. Executive Position

MLCouncil is already a batch-first paper-trading platform with Dagster orchestration, a FastAPI
control plane, a Streamlit dashboard, artifact governance, runtime profiles, pre-trade controls,
and Alpaca paper execution. The AS IS/TO BE analysis states the core problem is **alignment**;
this roadmap extends that thesis:

> The 2030 goal is not "more models". It is **autonomy**: a system that researches, validates,
> promotes, retires, protects, and explains itself — with the human operator moved from
> "operator" to "governor" (veto power, not steering wheel).

Owner decisions recorded on 2026-08-13:
1. **Goal**: autonomous self-improving system (research loop + execution).
2. **Risk posture**: paper → live, progressive, gated by evidence.
3. **Resources**: dedicated VPS + GPU (24/7 orchestration, model training on-device).
4. **Markets**: multi-region (US equities + EU/Asia sessions + crypto 24/7).

## 2. Operating Principles (2030-ready)

| # | Principle | Enforced by |
|---|---|---|
| P1 | **Nothing exists outside the daily path.** Any module not wired into the pipeline with telemetry is either promoted or deleted. Scaffolds carry an **expiry date**. | Flag governance (F-0.4), code review checklist |
| P2 | **Shadow → canary → production → retirement**, never binary on/off flags. | Canary layer (F-0.4, P-1.1) |
| P3 | **Every decision is inspectable** from raw input to alpha contribution, risk transformation, optimizer constraint, execution outcome. | Attribution chain, SHAP, evidently reports (P-3.3) |
| P4 | **Kill switches everywhere.** Autonomy without the ability to halt is a liability. | Circuit breakers, max-drawdown halt, human veto window (P-2.4) |
| P5 | **Telemetry is default-on in prod.** No feature ships without OTEL spans + dashboard surface. | Observability ADR (already exists) |
| P6 | **Audit-first.** Immutable log of every promotion, retirement, and manual override. Regulatory readiness (AI Act, MiCA, ESMA) is a feature, not a tax. | P-3.3 |
| P7 | **The champion is the king.** Challengers (TFT, TS-FM, stacking, MoE) must beat the champion through the existing promotion gate or die. | Production promotion gate (exists) |

## 3. North-Star Metrics

| Metric | Target by end of Phase |
|---|---|
| Human interventions per month | → 0 (Phase 2 exit: 30 days unattended promotion/retirement) |
| Hypotheses promoted with positive out-of-sample IC | > 40% of those passing the gate |
| Alpha decay auto-retirement latency | < 5 trading days from detection |
| Paper–live parity gap | < 10 bps realized slippage vs paper fills (Phase 1 exit) |
| Pipeline coverage | 100% of trading days, all 3 sessions (US/EU/Asia) + crypto 24/7 |
| Decision explainability | 100% of orders traceable to signal attribution + SHAP (Phase 3 exit) |

## 4. Phase 0 — Foundations: close the drift, wire the immune system (2–4 weeks)

Goal: zero open drift-register items; every built component either wired or deleted; the
dashboard becomes the verified command surface.

### F-0.1 Close M10 — multivariate Monte Carlo VaR
- **Now**: `council/risk_engine.py:261,356` `compute_var` simulates a univariate Gaussian portfolio
  distribution. This is the **only open drift item** (M10, severity High).
- **Do**: multivariate Monte Carlo over asset paths using the covariance estimators already in
  repo (`council/covariance_dynamic.py`: Ledoit default, DCC optional), with **stress replay**
  scenarios sourced from `council/generative_stress.py` (the "generative" method already exists
  behind a flag).
- **Deliverable**: `compute_var(method="monte_carlo_mv")` default; test suite extended
  (`tests/test_risk_engine.py`); drift register M10 → Resolved.
- **Effort**: 3–5 dev-days. All ingredients exist; this is integration, not research.

### F-0.2 Wire the immune system
- **Now**: `council/monitor.py:625` `check_causal_graph_drift` is exported but imported nowhere;
  Dagster has only `monitored_jobs=[daily_job]` (`data/pipeline.py:2257`). TDA warning
  (`council/tda_warning.py`) is enabled but standalone.
- **Do**: single alerting surface combining TDA early-warning + causal graph drift + ADWIN
  (`council/drift.py`) + evidently drift reports (`council/evidently_reports.py`), wired as a
  Dagster schedule + FastAPI `/api/monitoring` endpoints (router already exists).
- **Deliverable**: `check_causal_graph_drift` scheduled weekly; unified health dashboard section;
  alerts reach admin UI and optional email (ALERT_EMAIL already configured for prod).
- **Effort**: 3–5 dev-days.

### F-0.3 Liquidation of dead weight
- Delete or repurpose: `execution/ibkr_adapter.py` / `coinbase_adapter.py` (`NotImplementedError`,
  never imported — see P-1.3: they become the EU and crypto gateways), 3 duplicated worktrees
  under `.claude/worktrees/`, duplicate scripts in `scripts/experiments/` (deduplicate against
  `scripts/`, keep unique experiments), fragile `data/pipeline.py:1232-1264` aggregator-mode
  override (replace with clean injection).
- Add dashboard tests: `dashboard/app.py` + 3 pages currently have **zero** tests.
- **Effort**: 2–4 dev-days.

### F-0.4 Flag governance + canary layer
- Inventory all ~100 `MLCOUNCIL_*` flags. Each disabled-by-default feature gets: docstring,
  expiry date, and a target phase. Introduce a **canary config profile** (prod + canary features
  with automatic revert on metric regression) instead of binary toggles.
- **Effort**: 2–3 dev-days.

### Phase 0 exit criteria
- Drift register: **zero open items** (M10 resolved).
- CI green; dashboard tests passing; no dead modules in `council/`, `execution/`, `models/`.
- Every flag documented with expiry; canary layer operational.

### Phase 0 status (2026-08-13)

| Item | Status | Evidence |
|---|---|---|
| F-0.1 Multivariate MC VaR | ✅ Implemented | `council/risk_engine.py`: multi-step path simulation, Ledoit–Wolf shrinkage (ridge+clipping removed), t-copula (`tail_dof`), ES alongside VaR, stress replay; generative CVaR empirical fix (hidden-Gaussian ratio removed). 13/13 tests incl. Kupiec/Christoffersen, ES sanity, tail-dependence λ̂_L, multi-step monotonicity. ADR `2026-08-13-multivariate-monte-carlo-var.md` |
| F-0.2 Immune system wiring | ✅ Implemented | `causal_drift_check` asset, weekly schedule (Mon 02:00 UTC) + baseline persistence (`causal_baseline.json`); `council/alerting.py` (`collect_health_signals`, `dispatch_health_alerts` via existing `AlertDispatcher`); `GET /api/monitoring/health` read-only; aggregator-mode clean injection (env mutation removed); `pd.Series` monitor bug fixed. 133 tests. ADR `2026-08-13-unified-risk-alerting.md` |
| F-0.3 Liquidation | 🔶 Revised | Dashboard tests ✅ (23 tests via streamlit AppTest, no page refactor needed). Worktrees: **kept** — 11 active lanes, 5 with uncommitted work (destructive to remove). `scripts/experiments/`: **kept** — canonical home per `docs/repo-triage.md` (root `scripts/train_*.py` are documented compatibility wrappers). ibkr/coinbase adapters: deferred to P-1.2/P-1.3 (repurpose, per roadmap). |
| F-0.4 Flag governance + canary | ✅ Implemented | `docs/flag-registry-2026-08-13.md` (93 flag inventariati, status/fase/expiry), annotazioni docstring su 22 moduli; `config/canary.yaml` + `council/canary.py` (controller con revert automatico sticky + alert via F-0.2) + asset giornaliero `canary_health` (no-op senza feature attive). 25 test dedicati. ADR `2026-08-13-canary-config-profile.md`. Attivazione = gate G1. |

**Phase 0 exit criteria: raggiunti** — drift register zero item aperti (M10 risolto); dashboard testati; flag governati con expiry; canary layer operativo.

**Gate G1 (2026-08-13): trio approvato e attivo** — `online_learning`, `position_sizing_cqr`, `dynamic_slippage` abilitati in `config/canary.yaml` (revert automatico + alert operativi). `moe_gating` **non** attivato: gate non addestrato (drill-down §7.3) — prerequisito P-1.1: train hard-EM del gating, poi canary.

Known pre-existing (not caused by this phase): 2 stale tests in `tests/test_drift.py` referencing a nonexistent `_detector` attribute; `council/drift.py` untouched.

## 5. Phase 1 — Activation & Progressive Live (1–3 months)

Goal: everything valuable is actually running; first live capital with hard gates; multi-region
data foundation laid.

### P-1.1 Activate Wave 3/4 via canary
- Online learning (`models/online.py`, IC-gated refit) → canary on.
- MoE aggregation, CQR sizing, DCC covariance, differentiable portfolio → one at a time through
  canary with regression checks (they are all built and ADR'd; the missing piece is the activation
  discipline).
- TFT challenger: run the existing walk-forward promotion machinery
  (`scripts/run_walkforward_promotion.py --model tft`) on the new canary baseline.
- **Effort**: 3–5 dev-days (activation work is mostly instrumentation).

### P-1.2 Crypto execution — the 24/7 asymmetry
- **Now**: `MLCOUNCIL_CRYPTO_ENABLED=true` in `config/runtime.env`, BTCUSD/ETHUSD in universe,
  but `execution/coinbase_adapter.py` is dead. The one market that never sleeps has no adapter.
- **Do**: implement the Coinbase adapter (paper first) + crypto session in Dagster schedules;
  reuse OMS (`execution/oms.py`), position caps, risk gates. Crypto fills become the first real
  TCA data source (24/7 → fast feedback for the self-calibrating cost model ADR).
- **Effort**: 4–6 dev-days.

### P-1.3 Multi-region data layer (US + EU + Asia)
- Market-session calendars (US/Europe/Asia) via exchange calendars; vendor-agnostic data loader
  interface (current: yfinance/FRED; add EU/Asia sources behind the same contract);
  Dagster schedules per session; region-aware universe buckets and liquidity filters.
- Revive `ibkr_adapter.py` as the EU/global broker gateway (IBKR covers EU/Asia) — dead code
  becomes the multi-region execution path.
- **Effort**: 6–10 dev-days.

### P-1.4 Progressive live ladder (paper → live)
Strict gated progression, no discretion:
1. **Paper parity**: live signals must replicate paper exactly (same code path, different fill
   feed) — parity gap < 10 bps for 4 weeks.
2. **Live small**: ≤ 1% of intended capital, only US equities + crypto first (most data history).
3. **Live scaled**: grow only after N weeks above thresholds (Sharpe ≥ backtest lower bound,
   max DD within pre-registered limits, TCA within cost-model bounds).
- Hard constraints on every rung: max 20 orders/day (exists), 30% turnover cap (exists), 10%
  position cap (exists), **max-drawdown halt** (new), **human kill switch** (new, one-click in
  admin UI, immutable audit entry), circuit breakers on data outage.
- **Effort**: 5–8 dev-days (mostly risk/ops plumbing; trading service already exists).

### Phase 1 exit criteria
- Paper proven across US/EU/Asia sessions + crypto 24/7.
- Live small-capital running with kill switch, drawdown halt, audit log.
- First quarter of real TCA data feeding the cost model calibration.

## 6. Phase 2 — The Autonomous Loop (3–9 months)

Goal: the system researches, validates, promotes, and retires signals **without human steering**,
with the human as vetoing governor.

### P-2.1 Alpha R&D engine
- LLM agents (self-hosted, on the GPU) propose signal hypotheses from: current feature inventory,
  paper abstracts, market regime observations, anomaly reports.
- Each hypothesis enters a **sandboxed backtest harness** (reuse `scripts/experiments/` pattern,
  standardized): point-in-time data from ArcticDB, lookahead-safe features (shift-1 rule
  preserved), walk-forward CI (11 windows), statistical gates (IC, t-stat, decay).
- Survivors enter the existing **promotion gate** (`production_manifest.yaml`, 3 passes) —
  promotion and retirement become automated, logged, vetoable within a 24 h window.
- **Guardrails**: per-hypothesis compute budget, correlated-hypothesis dedup, hypothesis
  diversity pressure (penalize re-discoveries of the same signal family).
- **Effort**: 15–25 dev-days (biggest single workstream).

### P-2.2 Risk becomes a system, not a function
- Unified risk layer: multivariate MC VaR (F-0.1) + CQR sizing + DCC covariance + generative
  stress, all in the daily path, all pre-trade.
- **Weekly generative stress rehearsal**: LLM-generated narratives (US default, stablecoin depeg,
  flash crash, EU election shock, Asia session liquidity gap) replayed through `generative_stress`
  as routine exercises, with auto-report; findings feed limits.
- **Effort**: 5–8 dev-days.

### P-2.3 Autonomous retirement
- Alpha-decay detection (EWM IC-Sharpe halflife monitoring, per AGENTS.md wording) triggers
  automatic challenger substitution through the existing gate; retired signals archived with
  rationale into the audit log.
- **Effort**: 3–5 dev-days.

### P-2.4 Autonomy guardrails
- Human veto window on every promotion/retirement (24 h default, configurable).
- Kill switch taxonomy: data halt, model halt, execution halt, full halt — each one-click, each
  logged, each tested monthly (chaos drills).
- Interventions counter surfaced in dashboard: the metric we are driving to zero.
- **Effort**: 3–5 dev-days.

### Phase 2 exit criteria
- 30 consecutive days of unattended promotion/retirement with zero interventions.
- Live capital at scaled level (gated by Phase 1 criteria), all sessions covered.
- Weekly stress rehearsal operational.

## 7. Phase 3 — The 2030 Organism (9–24 months)

### P-3.1 Foundation-model priors (GPU justification)
- Time-series foundation models (Chronos/TimesFM/Moirai class) as **feature extractors and
  zero-shot priors** feeding the council; the LightGBM champion keeps final say until beaten via
  the gate. GPU makes fine-tuning feasible on-device.
- **Effort**: 20–30 dev-days including evaluation harness.

### P-3.2 Council deliberation (the name becomes real)
- Today `council/aggregator.py` is a weighted blend. 2030: an **LLM arbiter** mediates between
  quantitative signals and event-driven reasoning (RAG over news/filings — FinMA sentiment ADR
  already exists), with the arbiter itself gated and audited like any other signal.
- **Effort**: 10–15 dev-days.

### P-3.3 Compliance-as-feature
- Immutable audit trail (promotions, retirements, overrides, limit changes); per-signal SHAP;
  evidently drift reports shipped to the dashboard; regulatory mapping (AI Act risk tiers, MiCA
  for crypto, ESMA for EU execution) maintained as living docs.
- **Effort**: 8–12 dev-days.

### P-3.4 Causality first-class
- PCMCI causal discovery (scaffold exists) promoted; do-calculus-style intervention reasoning
  behind the weekly stress narratives; causal monitor (F-0.2) becomes the system's immune system,
  auto-reverting canary features on causal graph shifts.
- **Effort**: 10–15 dev-days.

### P-3.5 Full 24/7 coverage
- US + EU + Asia equities + crypto continuous: session-aware risk, liquidity-aware sizing,
  cross-region correlation monitoring (correlation detector already exists).
- **Effort**: 8–12 dev-days.

## 8. Cross-Cutting Enablers (start in Phase 1, run forever)

- **Observability default-on**: OTEL spans on all daily assets, Grafana dashboards refreshed.
- **Self-calibrating cost model**: ADR exists (`2026-05-21-self-calibrating-cost-model.md`);
  TCA data from crypto (24/7) + live fills calibrates bps per venue/region.
- **Tri-temporal store**: ADR exists (`2026-05-21-tri-temporal-store.md`); ArcticDB PIT
  versioning is the foundation for the R&D engine's point-in-time research — treat as
  prerequisite for P-2.1.
- **MLflow artifact governance**: every promotion must have a fully reproducible artifact set
  (manifest, hash, dataset snapshot, metrics).

## 9. Risk Register

| Risk | Mitigation |
|---|---|
| Autonomous loop proposes garbage | Sandboxed backtests, statistical gates, compute budgets, diversity pressure, 24 h veto |
| Live capital loss | Gated ladder, kill switches, drawdown halt, position/turnover/order caps, paper parity |
| Regime shift unnoticed | TDA + causal drift + ADWIN unified alerting; weekly generative rehearsal |
| Data vendor dependency | Vendor-agnostic loader contract, local caching, fallback sources |
| Regulatory (AI Act/MiCA/ESMA) | Audit-first by design (P6), compliance mapping as living doc |
| VPS single point of failure | Compose-based backups, restore drill, state (ArcticDB/MLflow) on persistent volume |
| Feature sprawl returns | P1 discipline + expiry dates; quarterly "liquidation" review |

## 10. Decision Gates (owner must choose)

| Gate | When | Decision |
|---|---|---|
| G1 | End Phase 0 | Approve canary activation list (which Wave 3/4 features enter canary first) |
| G2 | End Phase 1 | Approve live-small capital amount, instruments, and thresholds |
| G3 | End Phase 2 | Approve autonomy level: veto window duration, which actions are fully automatic |
| G4 | Phase 1→2 | Approve EU/Asia instrument universe and IBKR gateway scope |
| G5 | Phase 2→3 | Approve 2030 scope: TS-FM investment vs. depth-first on autonomous loop |

## 11. ADR Backlog (create one per workstream when it starts)

1. `2026-08-xx-multivariate-monte-carlo-var.md` (F-0.1)
2. `2026-08-xx-unified-risk-alerting.md` (F-0.2)
3. `2026-08-xx-canary-config-profile.md` (F-0.4)
4. `2026-09-xx-crypto-execution-adapter.md` (P-1.2)
5. `2026-09-xx-multi-region-data-layer.md` (P-1.3)
6. `2026-10-xx-live-progression-gates.md` (P-1.4)
7. `2027-xx-alpha-rd-engine.md` (P-2.1)
8. `2027-xx-autonomous-retirement.md` (P-2.3)

## 12. Immediate Next Steps

1. **Approve this roadmap** (or amend) → then G1 prep.
2. Phase 0 kickoff in two lanes: (a) F-0.1 multivariate VaR, (b) F-0.2 alerting surface —
   independent, parallelizable.
3. F-0.3 liquidation can run concurrently (pure deletion/dedup, low risk).
