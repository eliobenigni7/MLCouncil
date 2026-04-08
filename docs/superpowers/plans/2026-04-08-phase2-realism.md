# Phase 2 Realism Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make backtests and model promotion materially closer to paper-trading reality with a shared transaction cost model, net-of-cost metrics, and deterministic out-of-sample diagnostics.

**Architecture:** Introduce one reusable transaction cost contract consumed by both portfolio construction and backtest reporting. Build net and gross backtest metrics on top of that contract, persist the expanded metrics to MLflow, and tighten model promotion using deterministic walk-forward and PBO-style diagnostics rather than single-point in-sample scores.

**Tech Stack:** Python, pandas, Dagster-adjacent pipeline code, MLflow, existing retraining pipeline, pytest

---

## Chunk 1: Shared Cost Model

### Files
- Create: `council/transaction_costs.py`
- Modify: `council/portfolio.py`
- Test: `tests/test_backtest.py`

### Tasks
- [ ] Write failing tests for reusable turnover and transaction-cost estimation.
- [ ] Run the targeted tests to verify the failures are real.
- [ ] Implement the minimal shared cost model with weight- and notional-based helpers.
- [ ] Wire `PortfolioConstructor` to use the shared cost model.
- [ ] Re-run the targeted tests until green.

## Chunk 2: Net-Of-Cost Backtests

### Files
- Modify: `backtest/report.py`
- Modify: `backtest/runner.py`
- Modify: `council/mlflow_utils.py`
- Test: `tests/test_backtest.py`
- Test: `tests/test_mlflow_contract.py`

### Tasks
- [ ] Write failing tests for net-of-cost equity, gross/net stats, and MLflow logging.
- [ ] Run the targeted tests to confirm the failures.
- [ ] Implement net/gross reporting and propagate the richer stats through `BacktestResult`.
- [ ] Log the expanded backtest metrics to MLflow with the required lineage tags.
- [ ] Re-run the targeted tests until green.

## Chunk 3: Walk-Forward Diagnostics And Promotion Gate

### Files
- Create: `backtest/validation.py`
- Modify: `data/retraining.py`
- Modify: `council/mlflow_utils.py`
- Test: `tests/test_backtest_validation.py`
- Test: `tests/test_mlflow_contract.py`

### Tasks
- [ ] Write failing tests for deterministic walk-forward splits, PBO-style diagnostics, and stricter promotion requirements.
- [ ] Run the targeted tests to verify they fail for the right reasons.
- [ ] Implement the validation helpers and integrate them into the promotion gate contract.
- [ ] Re-run the targeted tests until green.

## Chunk 4: Documentation And Verification

### Files
- Create: `docs/fase2-realism.md`
- Modify: `README.md`

### Tasks
- [ ] Document the new cost-model contract, net/gross metrics, and validation workflow.
- [ ] Run the relevant targeted tests, then the full suite.
- [ ] Commit the completed phase.
