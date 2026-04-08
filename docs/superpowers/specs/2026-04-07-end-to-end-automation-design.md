# End-to-End Automation via Dagster - Design Spec

**Date:** 2026-04-07
**Status:** Draft

---

## 1. Overview

Automate the full daily MLCouncil operating flow so the system can run without manual intervention from data ingest to Alpaca Paper execution.

The existing repository already has the main building blocks:

- `data/pipeline.py` for daily orchestration in Dagster
- `council/*` for aggregation and portfolio construction
- `api/services/trading_service.py` for Alpaca Paper execution with safety guards
- `api/*` for control-plane visibility through the Admin UI

The missing piece is not new business logic. The missing piece is an operational layer that turns these components into one reliable, observable, resumable daily workflow.

### Goal

Every trading day, the platform should:

1. fetch data
2. compute features and signals
3. build the target portfolio
4. write orders
5. validate tradability
6. execute orders automatically on Alpaca Paper
7. store execution results
8. raise alerts when anything degrades or fails

### Non-Goal

This design does not introduce live trading. Automation applies only to Paper Trading.

---

## 2. Problem Statement

Today the system is operationally fragmented:

- Dagster generates the analytical outputs
- order execution is a separate manual action
- status is spread across parquet files, local logs, UI pages, and service responses
- retry and fallback behavior are only partially centralized

This creates friction in four places:

1. the operator has to trigger or supervise too many steps
2. failures are visible late rather than at the point of failure
3. there is no single run record for "what happened today"
4. paper trading is safer than live, but still too manual for routine use

The target state is a single operational flow with a clear run status and controlled degradation.

---

## 3. Recommended Architecture

Use Dagster as the primary orchestrator and keep FastAPI as the control plane.

### 3.1 Responsibilities

**Dagster**

- Owns scheduling
- Owns step orchestration and dependencies
- Owns retries
- Owns materialization and run history
- Owns final execution handoff to paper trading

**FastAPI Admin**

- Shows current automation status
- Shows last pipeline run, last trading execution, and current alerts
- Allows manual rerun, replay, and emergency pause
- Exposes read-only operational APIs to inspect what happened

**Trading Service**

- Reads generated orders
- Enforces hard trading safety rules
- Submits orders to Alpaca Paper
- Stores execution logs locally

### 3.2 Why Dagster First

Dagster is the strongest fit because the repo already models the analytical workflow as daily assets with retries and partitions. Trading is operationally the last stage of the same business process, not a separate app concern.

If orchestration moved into FastAPI, the project would duplicate logic that Dagster already handles better:

- ordering and dependency management
- run-level observability
- partition-aware reruns
- failure boundaries
- sensor and schedule support

---

## 4. Proposed Flow

### 4.1 New End-to-End Job

Introduce a job dedicated to automated daily execution, for example:

- `daily_trade_pipeline`

This job should cover:

1. `raw_ohlcv`
2. `raw_news`
3. `raw_macro`
4. `alpha158_features`
5. `sentiment_features`
6. `lgbm_signals`
7. `sentiment_signals`
8. `current_regime`
9. `council_signal`
10. `portfolio_weights`
11. `daily_orders`
12. `pretrade_checks` new
13. `paper_trade_execution` new
14. `posttrade_summary` new

### 4.2 High-Level Data Flow

```text
Market data / News / Macro
        ->
Dagster assets
        ->
Signals + regime detection
        ->
Council aggregation
        ->
Portfolio optimization
        ->
data/orders/{date}.parquet
        ->
Pre-trade validation
        ->
Alpaca Paper execution
        ->
data/paper_trades/{date}.json
        ->
Operational summary + alerts
```

### 4.3 Schedule

Keep the schedule inside Dagster and align it to the market/data readiness assumptions already present in the project.

Recommended behavior:

- run once per business day after market-close data is expected to be available
- allow manual replay for a specific partition date
- skip duplicate execution for the same date if execution already succeeded

---

## 5. New Components

### 5.1 `pretrade_checks` asset/op

Add a pre-trade validation step between `daily_orders` and `paper_trade_execution`.

Its responsibilities:

- confirm the order file exists for the partition date
- ensure the order count is below `MLCOUNCIL_MAX_DAILY_ORDERS`
- estimate turnover and compare against `MLCOUNCIL_MAX_TURNOVER`
- verify no target weight exceeds `MLCOUNCIL_MAX_POSITION_SIZE`
- verify Alpaca is configured in paper mode only
- verify account connectivity and buying power
- optionally detect stale inputs such as missing latest market rows

Output:

- structured validation result
- hard failures list
- soft warnings list
- normalized execution payload for the next step

### 5.2 `paper_trade_execution` asset/op

This step will call `TradingService.execute_orders(date)` for the partition date.

Its responsibilities:

- execute only once for a given date unless explicitly forced
- capture submitted, rejected, and liquidated orders
- persist raw execution result
- mark run outcome as `success`, `degraded`, or `failed`

### 5.3 `posttrade_summary` asset/op

This step creates a normalized operational summary for UI and monitoring consumption.

Suggested output fields:

- partition date
- pipeline status
- data freshness status
- number of signals generated
- number of orders generated
- number of orders submitted
- number of orders rejected
- number of liquidations
- fallback modes activated
- warnings
- errors
- execution timestamp

Persist this summary as a JSON artifact, for example:

- `data/operations/{date}.json`

This becomes the single daily truth record for operations.

### 5.4 Automation state service

Add a small service layer in `api/services/` to aggregate:

- latest Dagster run status
- latest order date
- latest trade execution date
- latest operational summary
- emergency pause flag

This keeps Admin UI logic thin.

---

## 6. Failure and Degradation Policy

The user preference is clear: the system should continue automatically whenever it can do so safely.

The design therefore distinguishes between hard failures and soft degradations.

### 6.1 Hard failures: stop execution

These must block paper trading:

- Alpaca is not in paper mode
- no order file exists for the target date
- order count exceeds configured hard limit
- buying power or account state is invalid
- generated weights violate hard position-size limits
- portfolio construction fails entirely
- order payload is malformed

### 6.2 Soft degradations: continue with warnings

These should not block execution if the rest of the system is healthy:

- news download returns empty
- sentiment model is unavailable and falls back to neutral values
- macro download is partial but still usable
- individual orders are rejected while the rest are valid
- monitoring/notification side effects fail

### 6.3 Degraded success state

The run should support a `degraded` terminal state for cases like:

- pipeline completed
- orders were generated
- execution happened
- but one or more non-critical fallbacks or partial rejections occurred

This is important because "not failed" is not the same thing as "fully healthy".

---

## 7. Idempotency and Replay

Automation must be safe to rerun.

### 7.1 Partition-based idempotency

Every trading day is keyed by date.

Rules:

- only one successful paper execution per date by default
- reruns of analytical steps are allowed
- reruns of execution require an explicit force flag or a "not yet executed" state

### 7.2 Execution markers

Use one of these markers to prevent duplicate trading:

- presence of a successful entry in `data/paper_trades/{date}.json`
- presence of `execution_status = success` in `data/operations/{date}.json`

Recommended rule:

- `paper_trade_execution` should short-circuit if the date is already marked successful

### 7.3 Replay mode

Support two replay modes:

- `recompute_only`: rerun pipeline and regenerate outputs without trading
- `force_execute`: rerun and execute again only when explicitly authorized

---

## 8. Safety Model

The existing TradingService already contains good base protections. The automation layer should formalize them.

### 8.1 Hard limits

Configured in `config/runtime.env` and loaded at runtime:

```env
MLCOUNCIL_AUTO_EXECUTE=true
MLCOUNCIL_AUTOMATION_ENABLED=true
MLCOUNCIL_AUTOMATION_PAUSED=false
MLCOUNCIL_MAX_DAILY_ORDERS=20
MLCOUNCIL_MAX_TURNOVER=0.30
MLCOUNCIL_MAX_POSITION_SIZE=0.10
```

### 8.2 Emergency pause

Add a global pause flag:

- `MLCOUNCIL_AUTOMATION_PAUSED=true`

When set:

- Dagster can still run the analytical pipeline
- `paper_trade_execution` is skipped
- Admin UI shows automation as paused

This is the main operator kill switch.

### 8.3 Paper-mode enforcement

Execution must fail fast if:

- Alpaca base URL is not the paper endpoint
- adapter mode is not `paper`

This guard should exist both in `TradingService` and in pre-trade checks.

---

## 9. Observability

Automation without observability becomes another black box. The system needs one place to inspect the day.

### 9.1 Daily operations artifact

Create:

- `data/operations/{date}.json`

Suggested shape:

```json
{
  "date": "2026-04-07",
  "pipeline_status": "success",
  "trade_status": "degraded",
  "auto_execute": true,
  "paused": false,
  "orders_generated": 12,
  "orders_submitted": 10,
  "orders_rejected": 2,
  "liquidations": 1,
  "fallbacks": ["sentiment-neutral-fallback"],
  "warnings": ["2 orders rejected due to position cap"],
  "errors": [],
  "started_at": "2026-04-07T21:30:01Z",
  "completed_at": "2026-04-07T21:34:49Z"
}
```

### 9.2 Admin UI

Add a compact automation view showing:

- automation enabled/paused
- last successful pipeline date
- last successful trade execution date
- last run terminal state
- active warnings
- quick links to latest orders and latest trade log

### 9.3 Alerts

Trigger alerts for:

- hard pipeline failure
- hard pre-trade failure
- paper execution failure
- repeated degraded runs
- stale data beyond expected threshold

Notification delivery can remain simple at first:

- local logs
- existing monitoring UI
- email only for critical failures if already configured

---

## 10. FastAPI Role After Automation

FastAPI remains important, but no longer as the orchestrator.

It should provide:

- `GET /api/automation/status`
- `GET /api/automation/runs/latest`
- `POST /api/automation/pause`
- `POST /api/automation/resume`
- `POST /api/automation/replay`

These endpoints should not rebuild orchestration logic. They should call small services that:

- inspect state
- modify flags
- request Dagster actions

This keeps the system layered cleanly.

---

## 11. File-Level Design Changes

### New files

- `docs/superpowers/specs/2026-04-07-end-to-end-automation-design.md`
- `api/services/automation_service.py`

### Likely modified files

- `data/pipeline.py`
- `api/main.py`
- `api/routers/pipeline.py`
- `api/routers/trading.py`
- `api/routers/monitoring.py`
- `api/templates/admin.html`
- `api/static/js/admin.js`
- `api/static/css/admin.css`
- `config/runtime.env`

### Data outputs

- `data/orders/{date}.parquet`
- `data/paper_trades/{date}.json`
- `data/operations/{date}.json`

---

## 12. Rollout Plan

Implement in phases to keep risk low.

### Phase 1 - Internal orchestration

- add `pretrade_checks`
- add `paper_trade_execution`
- add `posttrade_summary`
- keep auto-execution feature-flagged off by default

Success criteria:

- full end-to-end run works locally
- no duplicate execution on rerun
- operations artifact is written correctly

### Phase 2 - Control plane

- add automation status endpoints
- expose automation state in Admin UI
- add pause/resume behavior

Success criteria:

- operator can see last run and pause automation without editing files

### Phase 3 - Automatic execution

- enable scheduled automatic execution in Dagster
- turn on `MLCOUNCIL_AUTO_EXECUTE=true` in the intended environment

Success criteria:

- at least several consecutive business-day runs complete without manual intervention

---

## 13. Testing Strategy

No linter or type checker is configured, so validation should rely on targeted tests.

### Add tests for

- pre-trade hard failure cases
- soft degradation cases
- duplicate execution prevention
- successful order execution summary
- paused automation behavior
- automation status service responses

Suggested test files:

- `tests/test_trading_service.py`
- `tests/test_pipeline.py`
- `tests/test_api_pipeline.py`
- `tests/test_api_monitoring.py`
- new `tests/test_automation_service.py`

### Manual verification

Run locally:

1. trigger a partitioned Dagster run
2. verify `data/orders/{date}.parquet` is produced
3. verify `data/paper_trades/{date}.json` is produced or correctly skipped
4. verify `data/operations/{date}.json` summarizes the day
5. verify Admin UI reflects the same state

---

## 14. Risks

### Operational risk

Automatic paper execution can hide recurring partial failures if degraded states are not surfaced clearly.

Mitigation:

- explicit `degraded` state
- daily operations artifact
- visible warnings in Admin UI

### Data freshness risk

Running before upstream data is complete could create low-quality orders.

Mitigation:

- add freshness checks in `pretrade_checks`
- keep the schedule after expected data availability

### Duplicate execution risk

The same date could be traded twice after retries or manual reruns.

Mitigation:

- execution markers by date
- force flag required for re-execution

---

## 15. Recommendation

The project should move to a Dagster-first end-to-end automation model where paper trading is treated as the final stage of the daily pipeline, not as a separate manual operation.

This gives the cleanest operating model:

- one scheduler
- one run history
- one daily status record
- one control plane
- minimal manual work

The core product change is small in concept:

pipeline generation and paper execution become one workflow with clear safety boundaries.

That is the simplest path to making the whole system feel automatic instead of procedural.
