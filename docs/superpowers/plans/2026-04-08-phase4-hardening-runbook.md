# Phase 4 Hardening Runbook Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden runtime operations for paper trading by validating environment profiles, exposing operational health, adding tests for untested critical modules, and documenting daily runbook and promotion criteria.

**Architecture:** Extend the existing runtime and health surfaces instead of adding a parallel control plane. Keep the hardening work focused on runtime config validation, operational status visibility, and test coverage for `execution/alpaca_adapter.py` and `data/store/arctic_store.py`.

**Tech Stack:** Python, FastAPI, pytest, pandas, polars, pathlib, unittest.mock

---

## Chunk 1: Runtime Hardening

### Task 1: Add runtime profile validation helpers

**Files:**
- Modify: `runtime_env.py`
- Test: `tests/test_runtime_env.py`

- [ ] **Step 1: Write failing tests for runtime profile validation**
- [ ] **Step 2: Run `python -m pytest tests/test_runtime_env.py -v` and verify failure**
- [ ] **Step 3: Add supported profile metadata, required-key checks, and paper/dev validation helpers to `runtime_env.py`**
- [ ] **Step 4: Re-run `python -m pytest tests/test_runtime_env.py -v` and verify pass**

### Task 2: Surface runtime health and latest operations state

**Files:**
- Modify: `api/routers/health.py`
- Test: `tests/test_api_health.py`

- [ ] **Step 1: Write failing tests for runtime status and operations summary in health response**
- [ ] **Step 2: Run `python -m pytest tests/test_api_health.py -v` and verify failure**
- [ ] **Step 3: Add helpers for runtime validation status and latest operations artifact summary**
- [ ] **Step 4: Re-run `python -m pytest tests/test_api_health.py -v` and verify pass**

## Chunk 2: Critical Test Gaps

### Task 3: Add adapter tests without live Alpaca dependency

**Files:**
- Test: `tests/test_alpaca_adapter.py`
- Modify: `execution/alpaca_adapter.py` only if tests expose a real bug

- [ ] **Step 1: Write failing tests for env loading, paper-mode guard, latest price lookup, and order logging**
- [ ] **Step 2: Run `python -m pytest tests/test_alpaca_adapter.py -v` and verify failure**
- [ ] **Step 3: Fix only the minimal adapter behavior needed by the tests**
- [ ] **Step 4: Re-run `python -m pytest tests/test_alpaca_adapter.py -v` and verify pass**

### Task 4: Add Arctic store tests with a fake backend

**Files:**
- Test: `tests/test_arctic_store.py`
- Modify: `data/store/arctic_store.py` only if tests expose a real bug

- [ ] **Step 1: Write failing tests for write/read/list behavior using a fake Arctic library**
- [ ] **Step 2: Run `python -m pytest tests/test_arctic_store.py -v` and verify failure**
- [ ] **Step 3: Fix only the minimal store behavior needed by the tests**
- [ ] **Step 4: Re-run `python -m pytest tests/test_arctic_store.py -v` and verify pass**

## Chunk 3: Runbook And Promotion Criteria

### Task 5: Document the operational runbook and promotion checklist

**Files:**
- Create: `docs/paper-trading-runbook.md`
- Create: `docs/model-promotion-criteria.md`
- Modify: `README.md`
- Create: `docs/fase4-hardening.md`

- [ ] **Step 1: Write concise docs for daily operator workflow, kill switch usage, failure triage, and promotion gates**
- [ ] **Step 2: Link the new docs from `README.md`**
- [ ] **Step 3: Verify docs align with Fase 1-3 contracts already in code**

## Chunk 4: Final Verification

### Task 6: Run integrated verification

**Files:**
- No new files

- [ ] **Step 1: Run targeted tests for new runtime, health, adapter, and Arctic store coverage**
- [ ] **Step 2: Run `python -m pytest`**
- [ ] **Step 3: Run `python -c "from data.pipeline import defs; print(defs.get_job_def('daily_pipeline').name)"`**
- [ ] **Step 4: Commit with a Fase 4 hardening message**
