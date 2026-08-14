# MLCouncil Foundation Cleanup And TO BE Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the AS IS mismatches identified by the combined analysis, produce a clean quant baseline, and prepare the TO BE roadmap without redesigning the dashboard prematurely.

**Architecture:** Keep the first tranche corrective and auditable: docs/config/code alignment, risk math correction, artifact loading hardening, sentiment/council semantics, and cost-model honesty. Advanced model work starts only after a clean baseline is measured.

**Tech Stack:** Python, pytest, Dagster assets, pandas/polars, CVXPY, Streamlit docs only for now.

---

## Chunk 1: Documentation And Semantics

### Task 1: Reconcile docs with current AS IS

**Files:**
- Modify: `README.md`
- Modify: `AGENTS.md`
- Modify: `docs/fase2-realism.md`
- Modify: `docs/fase5-governance.md`

- [ ] Read `docs/architecture-as-is-to-be-2026-05-21.md`.
- [ ] Update ensemble wording: daily path uses LightGBM and sentiment signals; HMM supplies regime context unless code now proves otherwise.
- [ ] Update universe wording from `config/universe.yaml`.
- [ ] Replace "rolling 100-day IR" wording with the actual EWM IC-Sharpe behavior.
- [ ] Clarify daily inference vs offline target engineering.
- [ ] Clarify current transaction cost model wording.
- [ ] Run `python -m pytest tests/test_council.py -v`.
- [ ] Commit as `docs: reconcile as-is architecture`.

## Chunk 2: Risk And Artifact Integrity

### Task 2: Implement multivariate Monte Carlo VaR

**Files:**
- Modify: `council/risk_engine.py`
- Modify: `tests/test_risk_engine.py`

- [ ] Add a failing covariance-sensitivity test for Monte Carlo VaR.
- [ ] Implement multivariate scenario draws aligned to current weights.
- [ ] Regularize covariance deterministically.
- [ ] Preserve seeded reproducibility.
- [ ] Run `python -m pytest tests/test_risk_engine.py -v`.
- [ ] Commit as `fix: make monte carlo var multivariate`.

### Task 3: Enforce hash policy for critical pickle loads

**Files:**
- Modify: `data/pipeline.py`
- Modify: `models/regime.py`
- Modify: other checkpoint loaders found by search
- Modify: relevant tests

- [ ] Search for `pickle.load` in runtime modules.
- [ ] Add tests for missing, mismatched, and valid hash sidecars.
- [ ] Make critical loads fail closed.
- [ ] Update fixtures to write sidecars where needed.
- [ ] Run `python -m pytest tests/test_models.py tests/test_pipeline.py tests/test_artifact_governance.py -v`.
- [ ] Commit as `fix: enforce checkpoint hash validation`.

## Chunk 3: Signal Semantics

### Task 4: Apply sentiment source weighting

**Files:**
- Modify: `data/pipeline.py`
- Modify: `models/sentiment.py` or `data/ingest/news_processor.py` if helper reuse requires it
- Modify: `tests/test_sentiment.py`
- Modify: `tests/test_pipeline.py`

- [ ] Add a failing test where source metadata changes weighted sentiment output.
- [ ] Reuse existing source weighting helpers.
- [ ] Keep missing-source fallback stable.
- [ ] Run `python -m pytest tests/test_sentiment.py tests/test_pipeline.py -k "sentiment" -v`.
- [ ] Commit as `fix: apply sentiment source weighting`.

### Task 5: Resolve council orthogonality semantics

**Files:**
- Modify: `council/aggregator.py`
- Modify: `tests/test_council.py`
- Modify: `README.md` or architecture docs

- [ ] Choose confidence shrinkage or simplex projection.
- [ ] Add tests proving the chosen behavior.
- [ ] If keeping confidence shrinkage, expose/log effective weight sum.
- [ ] Run `python -m pytest tests/test_council.py -v`.
- [ ] Commit as `fix: document council orthogonality semantics`.

### Task 6: Fix feature inventory and Parkinson volatility semantics

**Files:**
- Modify: `data/features/alpha158.py`
- Modify: `tests/test_features.py`
- Modify: `README.md`

- [ ] Add a deterministic Parkinson feature test.
- [ ] Apply canonical scaling or document the feature as uncalibrated range-vol.
- [ ] Update actual feature inventory wording.
- [ ] Run `python -m pytest tests/test_features.py tests/test_pipeline.py -k "alpha158" -v`.
- [ ] Commit as `fix: align technical feature semantics`.

## Chunk 4: Baseline And Next Design

### Task 7: Produce clean baseline report

**Files:**
- Create: `docs/internal/baselines/YYYY-MM-DD-clean-baseline.md`

- [ ] Run existing walk-forward/backtest validation commands available in the repo.
- [ ] Capture git sha, config hash, Python version, data date range, and caveats.
- [ ] Include Sharpe, IC, max drawdown, turnover, cost, benchmark deltas, and regime breakdown.
- [ ] Run `python -m pytest tests/test_backtest_validation.py tests/test_retraining.py -v`.
- [ ] Commit as `docs: add clean baseline report`.

### Task 8: Start dashboard brainstorming

**Files:**
- No code edits in this task.

- [ ] Read `dashboard/app.py`, `dashboard/charts.py`, and `dashboard/data_loader.py`.
- [ ] Prepare current dashboard map and missing diagnostics list.
- [ ] Offer 3 directions: command center, quant lab notebook, agentic council cockpit.
- [ ] Ask the user which direction to explore first.
