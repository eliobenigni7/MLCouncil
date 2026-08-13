# MLCouncil

![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-blue.svg)
![Python: 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![Status: paper trading](https://img.shields.io/badge/status-paper%20trading-yellow)

MLCouncil is an end-to-end **multi-signal paper trading system** for US equities and crypto. The current daily path uses a 2-signal ensemble (LightGBM technical + FinBERT sentiment) with HMM regime labeling as context for the council aggregator; a CVXPY optimizer converts the resulting target weights into daily orders for Alpaca Paper Trading. The 2026–2030 strategy and its mathematical foundation are documented in [docs/roadmap-2026-2030-autonomous-council.md](docs/roadmap-2026-2030-autonomous-council.md) and [docs/math-drilldown-2026-2030-autonomous-council.md](docs/math-drilldown-2026-2030-autonomous-council.md).

---

## Table of Contents

- [How It Works](#how-it-works)
- [Alpha Models](#alpha-models)
- [Council Aggregation](#council-aggregation)
- [Portfolio Construction](#portfolio-construction)
- [Conformal Position Sizing](#conformal-position-sizing)
- [Monitoring, Alerts and Canary](#monitoring-alerts-and-canary)
- [Asset Universe](#asset-universe)
- [Expected Results and Performance Criteria](#expected-results-and-performance-criteria)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Run the System](#run-the-system)
- [Daily Operational Flow](#daily-operational-flow)
- [Key API Endpoints](#key-api-endpoints)
- [Project Structure](#project-structure)
- [Documentation](#documentation)
- [Observability](#observability)
- [Testing](#testing)
- [Contributing](#contributing)

---

## How It Works

The system runs a daily Dagster pipeline with eight stages, scheduled at 21:30 ET on weekdays. Its job is to load the latest checkpoints and perform inference, not retrain the models end-to-end:

```mermaid
flowchart TD
    S1["1 · Ingest<br/>yfinance · FRED · RSS news"] --> S2["2 · Features<br/>Alpha158-inspired · sentiment · macro"]
    S2 --> S3["3 · Inference<br/>LightGBM · FinBERT · HMM regime"]
    S3 --> S4["4 · Signals<br/>cross-sectional z-scores"]
    S4 --> S5["5 · Council<br/>regime weights × EWM IC-Sharpe · orthogonality"]
    S5 --> S6["6 · Sizing<br/>conformal intervals · uncertainty multiplier"]
    S6 --> S7["7 · Portfolio<br/>CVXPY mean-variance + hard constraints"]
    S7 --> S8["8 · Execution<br/>pre-trade checks · kill switch · Alpaca Paper"]
```

Daily inference does not compute training targets (`compute_targets` is used in training/backtesting flows, not in the daily Dagster inference path).

Feature versioning is tracked for point-in-time correctness, with historical retrieval and backtesting support handled by the feature store layer.

---

## Alpha Models

### Technical Model — LightGBM + Technical Feature Set

**File:** `models/technical.py`

The technical model uses a point-in-time OHLCV/macro feature set inspired by the Qlib Alpha158 family, computed via Polars and deliberately shifted 1 day to eliminate lookahead bias. The exact runtime inventory is defined by `data/features/alpha158.py` (~60 numeric factors via `alpha158_feature_count()`); do not assume the name means exactly 158 factors. `park_vol_20d` uses the canonical Parkinson variance scale `1/(4 ln 2)`.

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

The council combines active alpha signals in two stages. In the current daily pipeline, `lgbm` and `sentiment` are the active alpha signals; the HMM model supplies the current regime label used to select regime-conditional weights.

### 1. Regime-Conditional Base Weights

| Regime     | Raw LightGBM | Raw Sentiment | Effective daily start after active-signal normalization |
|------------|--------------|---------------|---------------------------------------------------------|
| Bull       | 50%          | 30%           | ~62.5% LightGBM / ~37.5% sentiment |
| Bear       | 40%          | 20%           | ~66.7% LightGBM / ~33.3% sentiment |
| Transition | 45%          | 25%           | ~64.3% LightGBM / ~35.7% sentiment |

The config still contains HMM weights for a fuller 3-signal council design, but the daily `council_signal` path should be read as a 2-signal ensemble unless an HMM alpha signal is explicitly added.

### 2. Adaptive Reweighting (after 30 days of history)

After 30 days of observed IC history, base weights are scaled by each model's **EWM IC-Sharpe** over recent history (halflife up to 20 days, bounded by the configured history window):

$$\mathrm{IC}_t = \rho_{\mathrm{Spearman}}(s_{t-1},\, r_t) \qquad \mathrm{ICSharpe}_t = \frac{\mathrm{EWM}_h(\mathrm{IC}_t)}{\mathrm{EWM}_h^{\mathrm{std}}(\mathrm{IC}_t)}\,\sqrt{252}$$

where $\mathrm{IC}_t$ is the cross-sectional Spearman correlation between the signal of day $t-1$ and the forward return of day $t$, and $h$ is the EWM halflife. Models with consistently negative IC-Sharpe are down-weighted toward their floor. Weight bounds are enforced after renormalization:

- **Floor:** 5% per active model
- **Ceiling:** 70% per model

**Orthogonality enforcement:** Pairwise rolling 60-day signal correlations are monitored. If any pair exceeds 0.70, the junior model is down-weighted by a factor of 0.5 to maintain portfolio diversification across alpha sources.

Every `aggregate()` call logs per-model weights and contributions to MLflow for attribution analysis.

---

## Portfolio Construction

**File:** `council/portfolio.py`

CVXPY solves a mean-variance optimization problem each day:

$$\max_{w} \; \alpha_{\text{eff}}'\, w - \tfrac{1}{2}\,\lambda_{\text{risk}}\, w'\Sigma w - \lambda_{\text{tc}}\,\mathrm{TC}(w)$$

with risk penalty $\lambda_{\text{risk}} = 1/\sigma_{\max}^2$ and transaction costs
$\mathrm{TC}(w) = \tfrac{1}{2}\,\|w - w_{\text{curr}}\|_1 \cdot \tfrac{\text{comm}+\text{slippage}}{10^4}$, subject to:

$$\sum_i w_i = b \quad \text{(budget, tier-dependent)} \qquad w_i \ge 0 \quad \text{(long-only)}$$

$$w_i \le \text{cap}_i \qquad \|w - w_{\text{curr}}\|_1 \le \text{max\_turnover} \qquad w'\Sigma w \le \sigma_{\max}^2$$

$$\sum_{i \in S} w_i \le \text{cap}_S \quad \text{(sector)} \qquad |w'\beta| \le 0.40 \quad \text{(beta neutrality)}$$

where $\sigma_{\max} = 30\%/\sqrt{252}$ (daily vol cap), per-position caps and budget come from the size-adaptive tiers, and the sector cap (35% base) is relaxed by `compute_effective_sector_cap()` when the active universe is narrow. All values are environment-configurable — see the auto-generated risk table.

**Transaction cost model:** currently a configurable heuristic. Runtime defaults are 3 bps slippage + 1 bps commission = 4 bps total (configurable via `MLCOUNCIL_SLIPPAGE_BPS` and `MLCOUNCIL_COMMISSION_BPS`), estimated on one-way turnover. This is not yet a realized-slippage calibrated impact model.

Post-processing: positions below 1% weight are zeroed and the remainder renormalized to satisfy the budget constraint.

The optimizer reads the current Alpaca paper portfolio as the rebalancing baseline. If the broker snapshot is unavailable, order generation fails closed — it does not assume an empty portfolio.

---

## Conformal Position Sizing

**File:** `council/conformal.py`

Before portfolio construction, each signal is scaled by a **conformal multiplier** derived from MAPIE Jackknife+ prediction intervals (80% coverage). The idea: when the model's uncertainty interval is wide, the position is reduced; when the interval is tight, it is expanded.

| Interval width | Confidence | Multiplier |
|----------------|------------|------------|
| Narrow         | High       | up to 1.8× |
| Wide           | Low        | down to 0.3× |

$$m_i = \exp\!\Bigl(1 - \frac{w_i}{\mathrm{median}(w)}\Bigr), \qquad m_i \leftarrow \mathrm{clip}(m_i,\, 0.3,\, 1.8)$$

where $w_i$ is the prediction-interval width for asset $i$.

Coverage of 80% (rather than 90%) tightens intervals and increases average multipliers, improving expected alpha capture. Empirical coverage (~85–90%) sits well above the conservative theoretical jackknife+ bound $1 - 2\alpha\,n/(n+1)$; the residual miss rate is acceptable because diversification across the configured universe limits individual tail exposure.

**References:** Angelopoulos & Bates (2023), *Conformal Risk Control*; MAPIE library (Jackknife+ method).

---

## Monitoring, Alerts and Canary

**Files:** `council/monitor.py`, `council/alerts.py`, `council/alerting.py`, `council/canary.py`

Four families of daily checks run automatically:

| Check | Trigger condition | Severity |
|---|---|---|
| Alpha decay | Rolling IC < 0.01 for 5+ consecutive days | CRITICAL |
| Feature drift | KS test: > 20% of top-10 SHAP features have p-value < 0.05 | WARNING |
| SHAP stability | Jaccard overlap of top-10 features vs 30-day baseline < 70% | WARNING |
| Regime change | HMM new state + transition probability > 0.70 | INFO |

CRITICAL alerts trigger email dispatch via `council/alerts.py`. All alert results are exposed at `GET /api/monitoring/alerts` and logged to MLflow as scalar metrics.

### Immune system (weekly)

- `causal_drift_check` Dagster asset runs **Mondays 02:00 UTC** (baseline persisted across runs) and writes `data/results/causal_drift_latest.json`.
- `GET /api/monitoring/health` aggregates five signal families — TDA early warning, causal graph drift, ADWIN/DDM streaming drift, evidently dataset drift — into `{level: ok|warn|alert}`; alerts are dispatched through the standard `AlertDispatcher` channels (logs, dashboard state, CRITICAL email).

### Canary activation (F-0.4)

Shadow features activate through `config/canary.yaml` + `council/canary.py` (run-policy env injection; operator env wins). The daily `canary_health` asset records same-day council metrics; a feature reverts automatically (sticky, with CRITICAL alert) when its metric stays below `floor` for `min_days` consecutive runs. Since gate G1 (2026-08-13) the active trio is: **online learning**, **CQR position sizing**, **dynamic slippage**; `moe_gating` stays off until the gating network is trained. Flag inventory with expiry dates: `docs/flag-registry-2026-08-13.md`.

```mermaid
flowchart LR
    CY["config/canary.yaml"] --> AP["CanaryController.apply()<br/>run-policy env injection"]
    AP --> DJ["daily job"]
    DJ --> CH["canary_health asset<br/>same-day council metrics"]
    CH --> RV{"metric < floor<br/>for min_days runs?"}
    RV -- no --> DJ
    RV -- yes --> VT["sticky revert + CRITICAL alert<br/>data/results/canary_state.json"]
```

---

## Asset Universe

**File:** `config/universe.yaml`

The tradable universe is configured in `config/universe.yaml`, organised in three buckets (large-cap equities, mid-cap equities, crypto). Each bucket carries its own weight cap to keep the portfolio constructor feasible without re-tuning per-ticker limits.

### Configured buckets (current)

| Bucket | Source | Count | Per-ticker cap |
|---|---|---|---|
| Large-cap equities | `universe.large_cap` | 26 | `universe.settings.max_large_cap_weight` = 8% |
| Mid-cap equities | `universe.mid_cap` | 6 | `universe.settings.max_mid_cap_weight` = 5% |
| Crypto | `crypto_universe.large_cap` | 2 (`BTCUSD`, `ETHUSD`) | `MLCOUNCIL_MAX_CRYPTO_POSITION_SIZE` = 20% |
| **Total** | | **34** | |

### Research vs. trading universe

- The **research universe** is the raw set of tickers with parquet OHLCV under `data/raw/ohlcv/`. A larger set may be present historically even if not currently configured.
- The **trading universe** for a given date is filtered through `load_universe_as_of()` in `data/pipeline.py`, which applies `config/universe_history.yaml` (added/removed dates per ticker) to avoid survivorship bias when backtesting.

To reload the live count programmatically:

```bash
python -c "from data.pipeline import load_universe_as_of; print(len(load_universe_as_of()))"
```

Sector coverage spans 11 GICS-style buckets (mapped in `data/features/sector_exposure.py`): Technology, Healthcare, Financials, Consumer Discretionary, Consumer Staples, Industrials, Energy, Utilities, Real Estate, Communication Services, Materials. Crypto sits in a dedicated bucket separate from equities for sector-cap accounting.

**Minimum liquidity threshold:** $1,000,000 average daily volume. Data scheduled at 21:30 ET in the America/New_York timezone with up to 2-day forward fill for gaps.

**Macro inputs (from FRED):** VIXCLS, DGS10 (10Y Treasury), DGS2 (2Y Treasury), S&P 500 with 21/63/252-day rolling windows.

---

## Expected Results and Performance Criteria

### Model Promotion Gates

A model candidate is promoted to production only if all of the following gates are green:

| Gate | Requirement |
|------|-------------|
| Out-of-sample Sharpe | `oos_sharpe ≥ champion − 0.1` (defensive) |
| Probability of Backtest Overfitting proxy | `pbo ≤ 0.50` |
| Walk-forward windows | `walk_forward_window_count ≥ 8` |
| Consecutive weekly passes | `≥ 3` (streak) before promotion |
| MLflow lineage | `pipeline_run_id`, `data_version`, `feature_version`, `model_version` all present |
| Metrics logged | `sharpe`, `max_drawdown`, `turnover`, `oos_sharpe`, `oos_max_drawdown`, `oos_turnover` |

For validation-depth monitoring, the daily inference path also tracks `equal_weight_sharpe_delta`, `equal_weight_cagr_delta`, and `regime_count` from walk-forward diagnostics.
When component signals are available, walk-forward diagnostics also expose `ablation_analysis` with marginal Sharpe contribution per component.

Candidates are rejected if gross/net metrics diverge implausibly from the estimated transaction costs, or if manual overrides were required to pass any gate.

### Portfolio Risk Targets (hard constraints enforced at runtime)

<!-- BEGIN risk-table -->

_Auto-generated by `scripts/generate_risk_doc.py` — do not edit by hand._

| Constraint | Default | Source |
|---|---|---|
| Max position per ticker | 12% | `MLCOUNCIL_MAX_POSITION_SIZE` |
| Min position floor | 1% | `(constant)` |
| Max one-way turnover | 30% | `MLCOUNCIL_MAX_TURNOVER` |
| Max annualised portfolio vol | 30% | `(constant)` |
| Sector cap (dynamic floor) | 35% | `(constant, may relax)` |
| Beta neutrality | True | `(constant)` |
| Max |portfolio beta| | 0.50 | `(constant)` |
| Commission | 1.0 bps | `MLCOUNCIL_COMMISSION_BPS` |
| Slippage (static lookup) | 3.0 bps | `MLCOUNCIL_SLIPPAGE_BPS` |
| Transaction-cost penalty λ | 1.00 | `(constant)` |
| Min signal strength (|z|) | 0.20 | `MLCOUNCIL_MIN_SIGNAL_STRENGTH` |
| Drawdown circuit breaker | 7% | `MLCOUNCIL_MAX_DRAWDOWN_PCT` |
| Crypto enabled | True | `MLCOUNCIL_CRYPTO_ENABLED` |
| Max crypto position | 20% | `MLCOUNCIL_MAX_CRYPTO_POSITION_SIZE` |
| Max crypto turnover | 40% | `MLCOUNCIL_MAX_CRYPTO_TURNOVER` |

Size-adaptive tiers (from `PortfolioConstructor._get_portfolio_tier`):

| Portfolio value | Positions | Max pos. | Max turnover | Budget |
|---|---|---|---|---|
| < $5 000 | 3 | 45% | 50% | 100% |
| $5 000 – $25 000 | 5 | 25% | 40% | 100% |
| $25 000 – $100 000 | 10 | 15% | 35% | 100% |
| ≥ $100 000 | 12 | 13% | 30% | 85% |

<!-- END risk-table -->

Long-only (no shorts in current scope). Sector cap may be relaxed by `compute_effective_sector_cap()` to keep the optimiser feasible when the active universe is narrow.

### Backtest Realism Parameters

| Parameter | Value |
|-----------|-------|
| Fill model | Next-open (order at EOD → fill at T+1 open) |
| Slippage | 3 bps probabilistic |
| Commission | 1 bps (default, configurable) |
| Total transaction cost | 4 bps per one-way trade (default) |
| Capital assumption | Long-only, fully invested |

### Alpha Decay Thresholds

| Metric | Alert threshold |
|--------|-----------------|
| Rolling IC (Information Coefficient) | < 0.01 sustained for 5+ days |
| SHAP feature Jaccard overlap | < 70% vs 30-day baseline |
| KS test feature drift (top-10 SHAP) | > 20% of features with p < 0.05 |

### Adaptive Weight Stability

The council's adaptive reweighting requires at least 30 days of IC history before it activates. The implementation computes an EWM IC-Sharpe over recent observations with halflife up to 20 days and a configured history window. No model weight falls below 5% or exceeds 70% after performance reweighting. Orthogonality downweighting uses confidence shrinkage: effective model weights are not renormalized afterward, so `weight_sum` may be below 1.0 by design; the combined signal is z-scored downstream. Attribution exports `effective_weight_sum` when shrinkage applies.

### What to Expect in Paper Trading

During normal (non-bear) market conditions with a liquid universe:
- **Orders per day:** typically 5–15 (≤ 20 enforced by kill switch)
- **Turnover:** typically 5–15% one-way per rebalance (≤ 30% hard cap)
- **Regime stability:** the HMM tends to stay in a single state for multiple weeks unless macro conditions shift sharply
- **Signal quality check:** if IC stays above 0.01 for the LightGBM model, alpha has not decayed; sentiment model IC is more variable and may trigger warnings in low-news periods

These are **design-level targets** from the constraint and monitoring setup. Live out-of-sample performance depends on realized alpha, market conditions, and execution quality — not guaranteed.

---

## Architecture

```mermaid
flowchart LR
    subgraph DATA["Data layer"]
        ING["ingest<br/>OHLCV · news · macro"] --> FEAT["features<br/>Alpha158-inspired · sentiment · sector"]
        FEAT --> STORE["ArcticDB · point-in-time"]
    end
    subgraph DEC["Decision layer"]
        MOD["models<br/>LightGBM · FinBERT · HMM"] --> COU["council<br/>aggregation · conformal sizing"]
        COU --> POR["portfolio<br/>CVXPY optimizer"]
    end
    subgraph OPS["Operational layer"]
        TRD["trading_service<br/>preflight · risk · orders"] --> ALP["Alpaca Paper API"]
        IMM["immune system + canary<br/>drift alerts · auto-revert"] -.->|"alert / revert"| COU
        POR --> ORD["data/orders · parquet + lineage"]
    end
    subgraph SURF["Operator surfaces"]
        API["FastAPI admin :8000"]
        DSH["Streamlit :8501"]
        DAG["Dagster :3000"]
        MLF["MLflow :5000"]
    end
    FEAT --> MOD
    STORE --> MOD
    POR --> TRD
    TRD --> API
    DAG -.-> ING
    DAG -.-> TRD
```

---

## Quick Start

### Python Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements_lock.txt # pinned snapshot; keep aligned with requirements.txt floors
# API / admin extras are already included via requirements.txt; install requirements_api.txt only for standalone extras.
```

### Environment Configuration

Copy `.env.example` to `.env` in the project root and fill in the real secrets:

```env
# Alpaca Paper Trading (required for order execution)
ALPACA_API_KEY=your_alpaca_key
ALPACA_SECRET_KEY=your_alpaca_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets
MLCOUNCIL_API_KEY=your_admin_api_key

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

For paper profile deployments, `MLCOUNCIL_API_KEY` is required to keep the admin/API surface secured.

**Dependency note:** The current runtime floor is `yfinance>=1.0` and the pinned snapshot uses `yfinance==1.0.0`. Keep `requirements_lock.txt` aligned with `requirements.txt` when updating market-data dependencies.

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
GET  /api/monitoring/health
```

---

## Project Structure

```text
MLCouncil/
├── api/                  FastAPI backend, Admin UI, service layer
├── backtest/             NautilusTrader backtest engine and walk-forward validation
├── config/               Runtime profiles, universe, regime weights, model config
├── council/              Aggregator, portfolio, conformal sizer, risk engine, monitor, alerts, alerting, canary
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

### Current Analysis and TO BE
- [docs/architecture-as-is-to-be-2026-05-21.md](docs/architecture-as-is-to-be-2026-05-21.md) — AS IS drift register, TO BE concept, and cleanup roadmap from the combined analysis
- [docs/agentic-prompts-2026-05-21.md](docs/agentic-prompts-2026-05-21.md) — Ordered prompt pack for agentic implementation of AS IS fixes and TO BE preparation
- [docs/superpowers/plans/2026-05-21-mlcouncil-foundation-to-be.md](docs/superpowers/plans/2026-05-21-mlcouncil-foundation-to-be.md) — Implementation plan for foundation cleanup and baseline measurement

### Strategy (2026–2030)
- [docs/roadmap-2026-2030-autonomous-council.md](docs/roadmap-2026-2030-autonomous-council.md) — Autonomous council roadmap: Phase 0 closed (2026-08), canary activation, live-progression gates
- [docs/math-drilldown-2026-2030-autonomous-council.md](docs/math-drilldown-2026-2030-autonomous-council.md) — Exact formulas in the code, mathematical critiques, rigorous upgrades and their verification statistics
- [docs/flag-registry-2026-08-13.md](docs/flag-registry-2026-08-13.md) — Inventory of all `MLCOUNCIL_*` flags with status, target phase and expiry dates

---

## Observability

Distributed tracing (Track T1.4) uses OpenTelemetry with Grafana Tempo. Tracing is **disabled by default**; enable only when the observability stack is running.

```bash
docker compose -f docker-compose.observability.yml up -d
```

| Service | URL |
|---------|-----|
| Grafana | http://localhost:3001 (admin / admin) |
| Tempo API | http://localhost:3200 |
| OTLP HTTP | http://localhost:4318 |
| Prometheus | http://localhost:9090 |

Run the pipeline with tracing:

```bash
export MLCOUNCIL_OTEL_ENABLED=true
export OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4318/v1/traces
export OTEL_SERVICE_NAME=mlcouncil-dagster
python scripts/run_pipeline.py --partition 2026-05-20
```

Spans are emitted on Dagster assets `raw_ohlcv`, `alpha158_features`, `lgbm_signals`, and `daily_orders` (tags: `mlcouncil.layer`, `dagster.partition`). See `docs/adr/2026-05-21-otel-grafana.md`.

---

## Testing

```bash
python -m pytest                                   # full suite
python -m pytest tests/test_tracing.py -v          # OTel no-op / pipeline import
python -m pytest tests/test_council.py -v          # council aggregator + portfolio
python -m pytest tests/test_api_health.py -v       # health endpoint
python -m pytest tests/test_trading_service.py -v  # trading service
python -m pytest tests/test_alpaca_adapter.py -v   # adapter (mocked)
python -m pytest tests/test_arctic_store.py -v     # feature store (fake backend)
python -m pytest tests/ -k "test_aggregator"       # single test by name
python -m pytest tests/test_canary.py -v           # canary controller (revert logic)

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

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, testing, code style, repository conventions and commit rules.

## License

GNU Affero General Public License v3.0 (AGPL-3.0). See [LICENSE](LICENSE).

> **AGPL notice:** if you run a modified version of MLCouncil as a network
> service (e.g. the trading API or a hosted signal engine), section 13 of the
> AGPL requires you to offer the corresponding source of your modified version
> to the users of that service.
