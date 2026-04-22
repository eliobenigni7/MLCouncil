# MLCouncil

MLCouncil is an end-to-end **multi-model ensemble paper trading system** for US equities and crypto. Three independent alpha models produce daily signals that a regime-aware council aggregator combines into portfolio weights, which a CVXPY optimizer converts into daily trading orders submitted to Alpaca Paper Trading.

---

## Table of Contents

- [How It Works](#how-it-works)
- [Alpha Models](#alpha-models)
- [Council Aggregation](#council-aggregation)
- [Portfolio Construction](#portfolio-construction)
- [Conformal Position Sizing](#conformal-position-sizing)
- [Monitoring and Alerts](#monitoring-and-alerts)
- [Asset Universe](#asset-universe)
- [Expected Results and Performance Criteria](#expected-results-and-performance-criteria)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Run the System](#run-the-system)
- [Daily Operational Flow](#daily-operational-flow)
- [Key API Endpoints](#key-api-endpoints)
- [Project Structure](#project-structure)
- [Documentation](#documentation)
- [Testing](#testing)

---

## How It Works

The system runs a daily pipeline with eight stages, orchestrated by Dagster and scheduled at 21:30 ET on weekdays:

```
Stage 1 — Ingest
  yfinance / FRED / RSS → raw OHLCV, macro series, news headlines

Stage 2 — Feature Engineering
  Alpha158 (158+ technical indicators, 1-day lookahead shift)
  FinBERT headline sentiment scores (SQLite-cached)
  Macro features: VIX, 10Y/2Y spread, S&P 500 rolling windows (21/63/252 days)

Stage 3 — Model Training / Refresh
  LightGBM technical model (CPCV cross-validation, SHAP logged to MLflow)
  FinBERT sentiment model (ProsusAI/finbert)
  3-state HMM regime detector (bull / bear / transition)

Stage 4 — Signal Generation
  Each model produces cross-sectional z-scores per ticker per day

Stage 5 — Council Aggregation
  Regime-conditional base weights scaled by each model's rolling 100-day IC-Sharpe
  Orthogonality enforcement: correlated models are automatically down-weighted
  Weight bounds: minimum 5%, maximum 70% per model

Stage 6 — Conformal Position Sizing
  MAPIE Jackknife+ prediction intervals (85% coverage)
  Multiplier range: [0.2 × signal, 2.0 × signal] based on model uncertainty
  Signals below confidence threshold are filtered before portfolio construction

Stage 7 — Portfolio Construction
  CVXPY mean-variance optimization with hard constraints (see below)
  Reads current Alpaca paper positions as rebalance baseline
  Output: data/orders/{date}.parquet with full lineage metadata

Stage 8 — Execution
  Pre-trade checks, kill switch, risk validation
  Notional-to-share conversion at current market price
  Submission to Alpaca Paper Trading API
  Artifacts: data/operations/, data/paper_trades/, data/risk/
```

All feature writes are versioned with `transaction_time` in ArcticDB (LMDB backend) to guarantee point-in-time correctness and prevent lookahead bias during backtesting.

---

## Alpha Models

### Technical Model — LightGBM + Alpha158

**File:** `models/technical.py`

The technical model uses over 158 point-in-time features derived from OHLCV data (inspired by the Qlib Alpha158 factor library), computed via Polars and deliberately shifted 1 day to eliminate lookahead bias.

**Training protocol:**
- Combinatorial Purged Cross-Validation (CPCV): dates split into 6 folds, all C(6,2) = 15 (train, test) combinations generated
- Embargo of 5 calendar days before each test fold to prevent leakage from overlapping forward returns
- One LightGBM per fold; best model (highest mean OOF IC) is selected for production
- SHAP feature importances logged to MLflow after each training run

**Key hyperparameters (from `config/models.yaml`):**
- Objective: regression on 5-day forward cross-sectional returns
- SHAP stability monitored: top-10 feature Jaccard overlap must stay ≥ 70% vs 30-day baseline

**References:** Marcos Lopez de Prado, *Advances in Financial Machine Learning* (CPCV purging/embargo); Qlib Alpha158 factor set.

---

### Sentiment Model — FinBERT

**File:** `models/sentiment.py`

Uses [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert), a BERT model fine-tuned on financial news, to score daily headlines per ticker. Scores are aggregated to a single daily sentiment z-score. Repeated headlines are cached in SQLite to avoid redundant inference.

**Signal:** Cross-sectional z-score of net positive sentiment per ticker.

---

### Regime Model — 3-State HMM

**File:** `models/regime.py`

A Gaussian Hidden Markov Model with 3 states trained on macro features (VIX, yield curve spread, S&P 500 rolling returns). The detected state — **bull**, **bear**, or **transition** — drives which base weights the council uses.

A regime change alert fires when the HMM emits a new state with transition probability > 0.70.

**References:** Hamilton (1989), *A New Approach to the Economic Analysis of Nonstationary Time Series*.

---

## Council Aggregation

**File:** `council/aggregator.py`

The council combines the three model signals in two stages:

### 1. Regime-Conditional Base Weights

| Regime     | LightGBM | Sentiment | HMM  |
|------------|----------|-----------|------|
| Bull       | 50%      | 30%       | 20%  |
| Bear       | 40%      | 20%       | 40%  |
| Transition | 45%      | 25%       | 30%  |

In bear regimes, HMM weight increases because the regime detector carries more information about market structure than the technical factor.

### 2. Adaptive Reweighting (after 30 days of history)

After 30 days of observed IC history, base weights are scaled by each model's **rolling 100-day Information Coefficient Sharpe** (mean IC / std IC × √252). Models with consistently negative IC-Sharpe are down-weighted toward their floor. Weight bounds are enforced after renormalization:

- **Floor:** 5% per active model
- **Ceiling:** 70% per model

**Orthogonality enforcement:** Pairwise rolling 60-day signal correlations are monitored. If any pair exceeds 0.70, the junior model is down-weighted by a factor of 0.5 to maintain portfolio diversification across alpha sources.

Every `aggregate()` call logs per-model weights and contributions to MLflow for attribution analysis.

---

## Portfolio Construction

**File:** `council/portfolio.py`

CVXPY solves a mean-variance optimization problem each day:

```
maximize   (α ⊙ conformal_multipliers)' w − tc_penalty × turnover
subject to
    Σ w_i    = 1          (fully invested)
    w_i     ≥ 0           (long-only)
    w_i     ≤ 10%         (per-position cap; 8% large-cap, 5% mid-cap)
    Σ |w_i − w_curr_i|   ≤ 30%   (one-way turnover cap)
    w' Σ w  ≤ (20%/√252)²        (daily volatility cap)
    sector[w] ≤ 25%              (sector exposure cap)
```

**Transaction cost model:** defaults are 3 bps slippage + 0 bps commission = 3 bps total (configurable via `MLCOUNCIL_SLIPPAGE_BPS` and `MLCOUNCIL_COMMISSION_BPS`), estimated on one-way turnover. Both gross and net equity curves are reported.

Post-processing: positions below 1% weight are zeroed and the remainder renormalized to satisfy the budget constraint.

The optimizer reads the current Alpaca paper portfolio as the rebalancing baseline. If the broker snapshot is unavailable, order generation fails closed — it does not assume an empty portfolio.

---

## Conformal Position Sizing

**File:** `council/conformal.py`

Before portfolio construction, each signal is scaled by a **conformal multiplier** derived from MAPIE Jackknife+ prediction intervals (85% coverage). The idea: when the model's uncertainty interval is wide, the position is reduced; when the interval is tight, it is expanded.

| Interval width | Confidence | Multiplier |
|----------------|------------|------------|
| Narrow         | High       | up to 2.0× |
| Wide           | Low        | down to 0.2× |

Coverage of 85% (rather than 90%) was chosen to tighten intervals and increase average multipliers by ~15%, improving expected alpha capture. The 15% miss rate is acceptable because diversification across 19 tickers limits individual tail exposure.

**References:** Angelopoulos & Bates (2023), *Conformal Risk Control*; MAPIE library (Jackknife+ method).

---

## Monitoring and Alerts

**Files:** `council/monitor.py`, `council/alerts.py`

Four families of daily checks run automatically:

| Check | Trigger condition | Severity |
|---|---|---|
| Alpha decay | Rolling IC < 0.01 for 5+ consecutive days | CRITICAL |
| Feature drift | KS test: > 20% of top-10 SHAP features have p-value < 0.05 | WARNING |
| SHAP stability | Jaccard overlap of top-10 features vs 30-day baseline < 70% | WARNING |
| Regime change | HMM new state + transition probability > 0.70 | INFO |

CRITICAL alerts trigger email dispatch via `council/alerts.py`. All alert results are exposed at `GET /api/monitoring/alerts` and logged to MLflow as scalar metrics.

---

## Asset Universe

**File:** `config/universe.yaml`

**19 equities across two segments:**

| Segment | Tickers | Max Weight/Position |
|---------|---------|---------------------|
| Large-cap (6) | AAPL, MSFT, GOOGL, AMZN, META, NVDA | 8% |
| Mid-cap (13) | ETSY, DOCU, UBER, ABNB, PLTR, SNOW, CRWD, NET, SQ, SHOP, FVRR, ROKU, DDOG | 5% |

**Crypto (in progress):**
| Tickers | Max Weight/Position |
|---------|---------------------|
| BTCUSD, ETHUSD | 20% (higher limit for volatility profile) |

**Minimum liquidity threshold:** $1,000,000 average daily volume. Data scheduled at 21:30 ET in the America/New_York timezone with up to 2-day forward fill for gaps.

**Macro inputs (from FRED):** VIXCLS, DGS10 (10Y Treasury), DGS2 (2Y Treasury), S&P 500 with 21/63/252-day rolling windows.

---

## Expected Results and Performance Criteria

### Model Promotion Gates

A model candidate is promoted to production only if all of the following gates are green:

| Gate | Requirement |
|------|-------------|
| Out-of-sample Sharpe | `oos_sharpe > 0` |
| Probability of Backtest Overfitting proxy | `pbo ≤ 0.50` |
| Walk-forward windows | `walk_forward_window_count ≥ 1` |
| MLflow lineage | `pipeline_run_id`, `data_version`, `feature_version`, `model_version` all present |
| Metrics logged | `sharpe`, `max_drawdown`, `turnover`, `oos_sharpe`, `oos_max_drawdown`, `oos_turnover` |

For validation-depth monitoring, retraining also tracks `equal_weight_sharpe_delta`, `equal_weight_cagr_delta`, and `regime_count` from walk-forward diagnostics.
When component signals are available, walk-forward diagnostics also expose `ablation_analysis` with marginal Sharpe contribution per component.

Candidates are rejected if gross/net metrics diverge implausibly from the estimated transaction costs, or if manual overrides were required to pass any gate.

### Portfolio Risk Targets (hard constraints enforced at runtime)

| Constraint | Limit |
|------------|-------|
| Max single position | 10% of portfolio (8% large-cap, 5% mid-cap) |
| Daily one-way turnover | ≤ 30% |
| Annualized portfolio volatility | ≤ 20% |
| Single sector exposure | ≤ 25% |
| Long-only | Yes (no shorts in current scope) |

### Backtest Realism Parameters

| Parameter | Value |
|-----------|-------|
| Fill model | Next-open (order at EOD → fill at T+1 open) |
| Slippage | 3 bps probabilistic |
| Commission | 0 bps (default, configurable) |
| Total transaction cost | 3 bps per one-way trade (default) |
| Capital assumption | Long-only, fully invested |

### Alpha Decay Thresholds

| Metric | Alert threshold |
|--------|-----------------|
| Rolling IC (Information Coefficient) | < 0.01 sustained for 5+ days |
| SHAP feature Jaccard overlap | < 70% vs 30-day baseline |
| KS test feature drift (top-10 SHAP) | > 20% of features with p < 0.05 |

### Adaptive Weight Stability

The council's adaptive reweighting requires at least 30 days of IC history before it activates. The rolling IC-Sharpe window is 100 days — chosen over shorter windows (60 days is considered too noisy for equity IC-Sharpe estimation, where noise dominates signal over short horizons). No model weight falls below 5% or exceeds 70% after renormalization.

### What to Expect in Paper Trading

During normal (non-bear) market conditions with a liquid universe:
- **Orders per day:** typically 5–15 (≤ 20 enforced by kill switch)
- **Turnover:** typically 5–15% one-way per rebalance (≤ 30% hard cap)
- **Regime stability:** the HMM tends to stay in a single state for multiple weeks unless macro conditions shift sharply
- **Signal quality check:** if IC stays above 0.01 for the LightGBM model, alpha has not decayed; sentiment model IC is more variable and may trigger warnings in low-news periods

These are **design-level targets** from the constraint and monitoring setup. Live out-of-sample performance depends on realized alpha, market conditions, and execution quality — not guaranteed.

---

## Architecture

```text
                              +----------------------+
                              |       MLflow         |
                              | runs, metrics, tags  |
                              +----------+-----------+
                                         ^
                                         |
+--------------+   +----------------+   +-----------------+   +-------------------+
| data/ingest  |-->| data/features  |-->| models + council|-->| council/portfolio |
| OHLCV/news/  |   | Alpha158,      |   | signals, regime |   | target weights    |
| macro (FRED) |   | sentiment,     |   | weights, council|   | data/orders/*.pq  |
+--------------+   | sector exposure|   +-----------------+   +---------+---------+
       |           +-------+--------+           |                       |
       v                   v                    v                       v
 +-----------+       +-----------+     +------------------+   +------------------+
 | raw data  |       | ArcticDB  |     | conformal sizer  |   | trading_service  |
 | parquet   |       | LMDB,     |     | MAPIE Jackknife+ |   | preflight, risk, |
 +-----------+       | point-in- |     +------------------+   | reconcile, submit|
                     | time vrsn |                             +---------+--------+
                     +-----+-----+                                       |
                           |                                             v
                     +-----+-----+                             +------------------+
                     |  Dagster  |                             | Alpaca Paper API |
                     | pipeline  |                             | orders & fills   |
                     | 21:30 ET  |                             +---------+--------+
                     +-----+-----+                                       |
                           +---------------------+------------------------+
                                                 |
                      +--------------------------+--------------------------+
                      |                                                     |
                      v                                                     v
             +------------------+                                 +--------------------+
             | FastAPI Admin UI |                                 | Streamlit Dashboard|
             | control plane    |                                 | read-only monitor  |
             | :8000            |                                 | :8501              |
             +------------------+                                 +--------------------+
```

---

## Quick Start

### Python Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements_api.txt   # for API / admin only
```

### Environment Configuration

Copy `.env.example` to `.env` in the project root and fill in the real secrets:

```env
# Alpaca Paper Trading (required for order execution)
ALPACA_API_KEY=your_alpaca_key
ALPACA_SECRET_KEY=your_alpaca_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Storage and tracking
ARCTICDB_URI=lmdb://data/arctic/
MLFLOW_TRACKING_URI=http://localhost:5000
DATABASE_URL=postgresql://mlcouncil:password@localhost:5432/mlcouncil

# Optional: alerts
ALERT_EMAIL=your@email.com
SMTP_PASSWORD=your_smtp_password

# Optional: market data enrichment
POLYGON_API_KEY=your_polygon_key

# Intraday runtime defaults
MLCOUNCIL_INTRADAY_AGENT_PROVIDER=rule-based
MLCOUNCIL_INTRADAY_LOG_TO_MLFLOW=false
```

**Runtime safety limits** (set these for paper trading):

```env
MLCOUNCIL_ENV_PROFILE=paper
MLCOUNCIL_MAX_DAILY_ORDERS=20
MLCOUNCIL_MAX_TURNOVER=0.30
MLCOUNCIL_MAX_POSITION_SIZE=0.10
MLCOUNCIL_AUTOMATION_PAUSED=false
MLCOUNCIL_AUTO_EXECUTE=false
```

Profile templates: `config/runtime.local.env.example`, `config/runtime.paper.env.example`.

### Docker Secrets

When running with Docker Compose, prefer Docker secrets for broker and market-data credentials:

```text
secrets/alpaca_api_key
secrets/alpaca_secret_key
secrets/polygon_api_key
secrets/smtp_password
```

The application reads `/run/secrets/*` first, then falls back to environment variables.

**Dependency note:** Keep `yfinance` on `0.2.x`. The repo pins `yfinance>=0.2.40,<1.0` for compatibility with `alpaca-trade-api`.

---

## Run the System

### Docker Compose (recommended)

```bash
docker compose build
docker compose up -d
```

| Service | URL |
|---------|-----|
| Admin UI + API | http://localhost:8000 |
| Streamlit Dashboard | http://localhost:8501 |
| Dagster UI | http://localhost:3000 |
| MLflow UI | http://localhost:5000 |

The Compose stack also starts an `intraday-supervisor` container. It auto-starts on boot and runs an intraday cycle every `MLCOUNCIL_INTRADAY_INTERVAL_MINUTES` during US market hours.

### Local (no Docker)

```bash
python run_admin.py                           # FastAPI admin API
streamlit run dashboard/app.py                # Public dashboard
dagster dev -f data/pipeline.py               # Dagster pipeline UI
python scripts/run_pipeline.py                # Standalone demo run
```

---

## Daily Operational Flow

### 1. Pre-Check

Before running:

```bash
GET /api/health           # runtime + trading_operations summary
GET /api/trading/status   # paused, kill_switch_active, paper_guard_ok
```

Abort if `paper_guard_ok=false`, `paused=true`, or any `HIGH` risk breach.

### 2. Pipeline Run

Trigger via Dagster UI or API:

```bash
curl -X POST http://localhost:8000/api/pipeline/run \
  -H "Content-Type: application/json" \
  -d '{"partition":"2026-04-11"}'
```

This produces:
- `data/orders/{date}.parquet` with lineage metadata
- Model artifacts and MLflow runs

### 3. Preflight + Execute

```bash
# Review preflight (blocks if any control fires)
curl http://localhost:8000/api/trading/preflight/2026-04-11

# Execute (only if preflight is green)
curl -X POST http://localhost:8000/api/trading/execute \
  -H "Content-Type: application/json" \
  -d '{"date":"2026-04-11"}'
```

**Hard stop conditions** — do not execute if:
- `pretrade.blocked=true`
- Any `HIGH` breach in `data/risk/risk_report_{date}.json`
- Projected turnover above limit
- Daily order count above `MLCOUNCIL_MAX_DAILY_ORDERS`

### 4. Post-Run Verification

Artifacts to inspect:

```
data/operations/{date}.json     # operational state (trade_status: success/degraded/blocked)
data/paper_trades/{date}.json   # submission log and liquidations
data/risk/risk_report_{date}.json  # projected portfolio risk
```

### Auto-Execute Mode

Set `MLCOUNCIL_AUTO_EXECUTE=true` to skip the manual execution step. After a successful Dagster run, the system automatically monitors completion and submits the orders through the trading service.

### Kill Switch

```env
MLCOUNCIL_AUTOMATION_PAUSED=true
```

This stops order execution but keeps the analytical pipeline running. `POST /api/trading/execute` returns `409` while paused. Reset to `false` only after resolving the underlying issue.

---

## Intraday Runtime

The intraday path is intentionally separate from the Dagster daily pipeline.

- `daily_pipeline` remains an end-of-day batch job.
- `intraday-supervisor` runs lightweight 15-minute cycles during market hours.
- By default the intraday decision engine is local `rule-based`, not OpenAI-backed.

Current intraday data path:

- Market snapshot: Alpaca intraday snapshot
- Historical daily enrichment: Polygon `/v2/aggs/ticker/{ticker}/prev`
- News enrichment: Polygon `/v2/reference/news`

This hybrid path is intentional. Many Polygon plans do not include the real-time stock snapshot endpoints used by higher-tier integrations. The adapter therefore avoids unsupported Polygon endpoints and degrades gracefully to Alpaca market data while still using Polygon where the key is entitled.

Manual controls:

```text
POST /api/intraday/control/start
POST /api/intraday/control/pause
POST /api/intraday/control/resume
POST /api/intraday/control/stop
POST /api/intraday/cycle
GET  /api/intraday/status
GET  /api/intraday/decisions/latest
POST /api/intraday/decisions/{decision_id}/execute
```

Key settings:

```env
MLCOUNCIL_INTRADAY_INTERVAL_MINUTES=15
MLCOUNCIL_INTRADAY_UNIVERSE=AAPL,MSFT,NVDA,AMZN,META,GOOGL,TSLA
MLCOUNCIL_INTRADAY_AGENT_PROVIDER=rule-based
MLCOUNCIL_INTRADAY_LOG_TO_MLFLOW=false
```

---

## Key API Endpoints

### Health and Runtime

```
GET  /api/health
GET  /api/health/dagster
GET  /api/trading/status
```

### Pipeline

```
POST /api/pipeline/run
GET  /api/pipeline/status
GET  /api/pipeline/automation/{run_id}
```

### Trading

```
GET  /api/trading/orders/latest
GET  /api/trading/orders/pending/{date}
GET  /api/trading/preflight/{date}
GET  /api/trading/reconcile/{date}
POST /api/trading/execute
POST /api/trading/liquidate
GET  /api/trading/history
```

### Intraday

```
GET  /api/intraday/status
POST /api/intraday/control/start
POST /api/intraday/control/pause
POST /api/intraday/control/resume
POST /api/intraday/control/stop
POST /api/intraday/cycle
GET  /api/intraday/decisions/latest
GET  /api/intraday/decisions/{decision_id}/explain
POST /api/intraday/decisions/{decision_id}/execute
```

### Configuration

```
GET  /api/config/universe
PUT  /api/config/universe
GET  /api/config/regime-weights
PUT  /api/config/regime-weights
```

### Monitoring

```
GET  /api/monitoring/alerts
GET  /api/monitoring/alerts/history
```

---

## Project Structure

```text
MLCouncil/
├── api/                  FastAPI backend, Admin UI, service layer
├── backtest/             NautilusTrader backtest engine and walk-forward validation
├── config/               Runtime profiles, universe, regime weights, model config
├── council/              Aggregator, portfolio, conformal sizer, risk engine, monitor, alerts
├── dashboard/            Streamlit read-only dashboard
├── data/                 Ingestion, Alpha158 features, ArcticDB store, Dagster pipeline
├── docs/                 Phase docs, runbooks, promotion criteria, plans
├── execution/            Alpaca adapter, OMS
├── models/               LightGBM technical, FinBERT sentiment, HMM regime
├── scripts/              Utility and support scripts
├── tests/                Pytest suite (council, API, adapter, Arctic store, runtime env)
├── docker-compose.yml    Local multi-service stack
├── requirements.txt      Core dependencies
├── requirements_api.txt  API/admin extra dependencies
└── run_admin.py          Admin server entry point
```

---

## Documentation

### Phase Architecture Docs
- [docs/fase1-foundations.md](docs/fase1-foundations.md) — Data contracts, lineage, MLflow conventions
- [docs/fase2-realism.md](docs/fase2-realism.md) — Transaction cost model, gross/net metrics, walk-forward + PBO gate
- [docs/fase3-operational-controls.md](docs/fase3-operational-controls.md) — Pre-trade controls, kill switch, risk artifacts
- [docs/fase4-hardening.md](docs/fase4-hardening.md) — Runtime profile validation, health surface, test coverage
- [docs/fase5-governance.md](docs/fase5-governance.md) — Artifact manifests, expanded contracts, review/process governance, safer operator UX

### Operations
- [docs/paper-trading-runbook.md](docs/paper-trading-runbook.md) — Daily operator workflow, triage guide
- [docs/model-promotion-criteria.md](docs/model-promotion-criteria.md) — Promotion gates and qualitative checklist
- [docs/adr/README.md](docs/adr/README.md) — ADR workflow and template for major design/process decisions

---

## Testing

```bash
python -m pytest                                   # full suite
python -m pytest tests/test_council.py -v          # council aggregator + portfolio
python -m pytest tests/test_api_health.py -v       # health endpoint
python -m pytest tests/test_trading_service.py -v  # trading service
python -m pytest tests/test_alpaca_adapter.py -v   # adapter (mocked)
python -m pytest tests/test_arctic_store.py -v     # feature store (fake backend)
python -m pytest tests/ -k "test_aggregator"       # single test by name

# Phase 4 quality gates (incremental scope)
python -m pytest --cov=. --cov-report=term --cov-fail-under=68
python -m ruff check api/main.py api/auth.py api/services/trading_service.py runtime_env.py council/portfolio.py
python -m mypy --config-file mypy.ini
python -m pip_audit -r requirements.txt --progress-spinner off
python -m bandit -q -r api council execution runtime_env.py -lll
```

`tests/conftest.py` installs a `slowapi` stub so rate-limiting tests run without the package installed.

---

## Current Scope

The current production target is robust **paper trading on US equities** via Alpaca Paper, with crypto (BTC/USD, ETH/USD) support in progress. Kubernetes, GitOps, and live trading are intentionally out of scope until the paper-trading path is stable end to end.

---

## License

MIT License. See `LICENSE`.
