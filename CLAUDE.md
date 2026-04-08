# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

MLCouncil is a multi-model ensemble trading system. Three independent alpha models (Technical/LightGBM, Sentiment/FinBERT, Regime/HMM) produce signals that a weighted council aggregator combines into portfolio weights, which a CVXPY optimizer converts into daily trading orders.

## Commands

### Setup
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements_api.txt   # for API/admin only
```

### Run Services
```bash
python run_admin.py                              # FastAPI admin API at :8000
streamlit run dashboard/app.py                   # Public dashboard at :8501
dagster dev -f data/pipeline.py                  # Dagster pipeline UI at :3000
python scripts/run_pipeline.py                   # Standalone demo run
docker-compose up                                # Full stack (API + Dashboard + Dagster + MLflow)
```

### Tests
```bash
python -m pytest                                 # All tests
python -m pytest tests/test_council.py -v        # Single file
python -m pytest tests/test_api_health.py -v     # API tests
python -m pytest tests/ -k "test_aggregator"     # Single test by name
```

`tests/conftest.py` installs a `slowapi` stub so rate-limiting tests run without the package installed.

## Architecture & Data Flow

```
yfinance/FRED/RSS → data/ingest/ → data/features/ → models/ → council/ → execution/
                                          ↕
                                    ArcticDB (LMDB)
```

**Dagster** (`data/pipeline.py`) orchestrates four layers:
1. **Ingest**: `raw_ohlcv`, `raw_news`, `raw_macro` — pulls from yfinance, RSS feeds, FRED
2. **Features**: `alpha158_features` (158+ technical indicators via Polars), `sentiment_features` (FinBERT)
3. **Signals**: `lgbm_signals`, `sentiment_signals`, `current_regime` (3-state HMM)
4. **Council**: `council_signal` → `portfolio_weights` → `daily_orders` (parquet)

Scheduled at 21:30 ET weekdays with up to 2 retries per asset.

**Feature Store**: `data/store/arctic_store.py` wraps ArcticDB with point-in-time correctness via `transaction_time` versioning — critical for backtesting without lookahead bias. Alpha158 features are deliberately shifted 1 day to prevent lookahead.

## Key Component Interactions

### Council Aggregator (`council/aggregator.py`)
- Maintains regime-conditional base weights (bull/bear/transition configs from `config/universe.yaml`)
- Adapts weights based on rolling 60-day Information Ratio per model
- Enforces weight bounds: min 5%, max 70% per model with orthogonality constraints
- Logs per-model attribution to MLflow for each date

### Portfolio Constructor (`council/portfolio.py`)
- CVXPY mean-variance optimization
- Hard constraints: long-only, ≤10% per position, ≤30% turnover, ≤20% vol ceiling
- Sector exposure caps from `data/features/sector_exposure.py`
- Output: `data/orders/{date}.parquet`

### Conformal Position Sizing (`council/conformal.py`)
- MAPIE Jackknife+ prediction intervals for model uncertainty
- Scales position size down when model confidence is low
- Filters signals below confidence threshold before passing to portfolio

### Monitoring (`council/monitor.py`, `council/alerts.py`)
- IC < 0.01 for 5+ consecutive days → alpha decay alert
- KS test on SHAP features → drift alert
- SHAP feature Jaccard overlap < 70% → stability alert
- Email dispatch via `council/alerts.py` for CRITICAL severity

## API Structure (`api/`)

FastAPI app created in `api/main.py` with:
- API key validation middleware (header: `X-API-Key`)
- Rate limiting via `slowapi`
- 5 routers: `health`, `pipeline`, `portfolio`, `config`, `monitoring`
- Admin UI served at `/` via Jinja2 templates + static files

## Models

| Model | File | Key Detail |
|-------|------|------------|
| Technical | `models/technical.py` | LightGBM + CPCV cross-validation, SHAP logging to MLflow |
| Sentiment | `models/sentiment.py` | ProsusAI/finbert, SQLite cache for repeated headlines |
| Regime | `models/regime.py` | 3-state Gaussian HMM (bull/bear/transition) |

## Backtest & Execution

- **Backtest** (`backtest/runner.py`): NautilusTrader integration, next-open fill model, 3 bps slippage, 1 bps commission
- **Execution** (`execution/alpaca_adapter.py`): Alpaca paper/live trading adapter; `execution/oms.py` handles order management

## Environment Variables

Required in `.env` (or `config/runtime.env`):
```
ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL
POLYGON_API_KEY
ALERT_EMAIL, SMTP_PASSWORD
ARCTICDB_URI=lmdb://data/arctic/
MLFLOW_TRACKING_URI=http://localhost:5000
DATABASE_URL=postgresql://mlcouncil:password@localhost:5432/mlcouncil
```

## Asset Universe

Defined in `config/universe.yaml`: 6 large-cap + 13 mid-cap tickers. Concentration limits and liquidity thresholds are also set there and enforced in the portfolio constructor.
