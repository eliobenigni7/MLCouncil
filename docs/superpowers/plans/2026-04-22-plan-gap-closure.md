# Plan Gap Closure Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the remaining plan gaps in reproducibility, retraining/registry, operator UX, and minimal supporting docs.

**Architecture:** Keep changes additive and minimal. Reuse existing validation, manifest, health, and admin UI flows instead of introducing new subsystems. Implement the missing retraining module as a focused root module that depends on existing walk-forward utilities and artifact governance helpers.

**Tech Stack:** Python, FastAPI, pytest, existing admin JS, existing manifest utilities

---

## Chunk 1: Reproducibility metadata and locking baseline

### Task 1: Add failing tests for config fingerprint metadata

**Files:**
- Modify: `tests/test_artifact_governance.py`
- Modify: `runtime_env.py`
- Modify: `council/artifacts.py`

- [ ] Add a failing test asserting artifact manifests include `config_hash`.
- [ ] Run `python -m pytest tests/test_artifact_governance.py -v` and confirm failure.
- [ ] Implement a stable config fingerprint helper in `runtime_env.py`.
- [ ] Add `config_hash` to artifact manifests.
- [ ] Re-run `python -m pytest tests/test_artifact_governance.py -v` and confirm pass.

### Task 2: Add deterministic lock baseline

**Files:**
- Create: `requirements_lock.txt`
- Modify: `README.md`

- [ ] Add a pinned lock-style dependency file matching the current dependency layout.
- [ ] Update docs with the intended usage.

## Chunk 2: Retraining / registry root implementation

### Task 3: Make retraining tests fail in the expected way

**Files:**
- Create: `data/retraining.py`
- Test: `tests/test_retraining.py`

- [ ] Run `python -m pytest tests/test_retraining.py -v` and confirm failure because the module is missing.
- [ ] Implement minimal `ValidationResult`, `ModelValidator`, `ModelRegistry`, and `RetrainingPipeline`.
- [ ] Reuse `backtest.validation.build_walk_forward_splits` and `summarize_walk_forward_metrics`.
- [ ] Ensure `ModelRegistry.save_model()` writes a manifest sidecar.
- [ ] Re-run `python -m pytest tests/test_retraining.py -v` and confirm pass.

## Chunk 3: Health/API/admin UX for validation summary and inconsistency signals

### Task 4: Add failing API tests for health summary payload

**Files:**
- Create: `tests/test_api_health_summary.py`
- Modify: `api/routers/health.py`
- Modify: `api/static/js/admin.js`
- Modify: `api/templates/admin.html`
- Modify: `api/services/monitoring_service.py`

- [ ] Add a failing API test asserting `/api/health` includes validation/backtest summary and inconsistency summary.
- [ ] Run `python -m pytest tests/test_api_health_summary.py -v` and confirm failure.
- [ ] Implement summary readers in `api/routers/health.py`.
- [ ] Surface the new data in the overview/config admin UI.
- [ ] Re-run `python -m pytest tests/test_api_health_summary.py -v` and confirm pass.

## Chunk 4: Verification

### Task 5: Run focused regression suite

**Files:**
- Test: `tests/test_artifact_governance.py`
- Test: `tests/test_retraining.py`
- Test: `tests/test_api_health_summary.py`
- Test: `tests/test_api_trading.py`

- [ ] Run `python -m pytest tests/test_artifact_governance.py tests/test_retraining.py tests/test_api_health_summary.py tests/test_api_trading.py -v`.
- [ ] Fix any regressions.
- [ ] Summarize what changed and any residual follow-ups.
