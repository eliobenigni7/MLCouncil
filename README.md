# MLCouncil

A multi-model ensemble trading system with regime-conditional adaptive weighting, conformal position sizing, and comprehensive monitoring.

## Overview

MLCouncil implements a **council of alpha models** that combine signals from:
- **Technical Model**: LightGBM with Alpha158 features and CPCV cross-validation
- **Sentiment Model**: FinBERT-based financial news sentiment with recency decay
- **Regime Model**: Gaussian HMM for market state detection (bull/bear/transition)

The system aggregates these signals using **regime-conditional weights** that adapt based on rolling performance metrics, then optimizes portfolio allocations via mean-variance optimization with uncertainty-aware position sizing.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MLCouncil System Architecture                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐          │
│  │  Dagster     │────▶│  Council     │────▶│  Portfolio   │          │
│  │  Pipeline    │     │  Aggregator  │     │  Constructor │          │
│  └──────────────┘     └──────────────┘     └──────────────┘          │
│         │                    │                    │                  │
│         ▼                    ▼                    ▼                  │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐          │
│  │ Data Ingest  │     │  Monitoring  │     │  Backtest    │          │
│  │ (yfinance)   │     │  & Alerts    │     │  Engine      │          │
│  └──────────────┘     └──────────────┘     └──────────────┘          │
│                                                                      │
├─────────────────────────────────────────────────────────────────────┤
│                          Frontends                                   │
│  ┌──────────────────────┐     ┌──────────────────────┐              │
│  │  Streamlit Dashboard  │     │   FastAPI Admin UI   │              │
│  │  (Public, Read-only)  │     │   (Private, Full)    │              │
│  └──────────────────────┘     └──────────────────────┘              │
└─────────────────────────────────────────────────────────────────────┘
```

## Features

### Alpha Models
- **LightGBM Technical Model**: 158+ technical indicators with Combinatorial Purged Cross-Validation (CPCV)
- **FinBERT Sentiment Model**: Financial news sentiment with source credibility weighting and recency decay
- **HMM Regime Detection**: 3-state Gaussian Hidden Markov Model for market regime classification

### Council System
- Regime-conditional base weights (bull/bear/transition configurations)
- Adaptive weighting based on rolling 60-day Sharpe ratio
- Weight bounds: min 5%, max 70% per model
- Full attribution logging for performance analysis

### Risk Management
- Mean-variance portfolio optimization via CVXPY
- Constraints: long-only, max 10% per position, 30% turnover cap, 20% vol ceiling
- Conformal position sizing using MAPIE Jackknife+ intervals
- Low-confidence signal filtering

### Monitoring & Alerts
- Alpha decay detection (IC < 0.01 for 5+ consecutive days)
- Feature drift detection via KS test on SHAP features
- SHAP stability monitoring (Jaccard overlap < 70%)
- Regime change alerts
- Email notifications for CRITICAL issues

### Dashboards
- **Public Dashboard** (Streamlit): Performance metrics, model attribution, regime timeline
- **Admin Dashboard** (FastAPI): Pipeline control, portfolio management, config editing, alert monitoring

## Installation

### Requirements
- Python 3.10+
- PostgreSQL (optional, for production)
- MLflow server (optional, for experiment tracking)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/MLCouncil.git
cd MLCouncil

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root:

```env
# Trading API
ALPACA_API_KEY=your_alpaca_key
ALPACA_SECRET_KEY=your_alpaca_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Market Data
POLYGON_API_KEY=your_polygon_key

# Alerts (Gmail SMTP)
ALERT_EMAIL=your_email@gmail.com
SMTP_PASSWORD=your_app_password

# Storage
ARCTICDB_URI=lmdb://data/arctic/
MLFLOW_TRACKING_URI=http://localhost:5000
DATABASE_URL=postgresql://mlcouncil:password@localhost:5432/mlcouncil
```

## Usage

## Fase 1 Foundations

- Dagster ora espone asset checks bloccanti per i contratti di `raw_ohlcv`, `raw_news`, `raw_macro`, `alpha158_features`, `sentiment_features` e `daily_orders`.
- Gli ordini salvati in `data/orders/*.parquet` includono lineage minimo (`pipeline_run_id`, `data_version`, `feature_version`, `model_version`).
- MLflow usa tag standardizzati per training, retraining e backtest.
- La baseline CI vive in [`.github/workflows/ci.yml`](./.github/workflows/ci.yml).
- Le convenzioni operative di Fase 1 sono riepilogate in [`docs/fase1-foundations.md`](./docs/fase1-foundations.md).

## Fase 2 Realism

- I costi di transazione usano un contratto condiviso in `council/transaction_costs.py`.
- Il backtest usa l'equity netta come default operativo ma conserva anche la curva lorda.
- Il runner e MLflow espongono metriche `gross` e `net`, inclusi `estimated_costs_usd`, `gross_final_equity` e `net_final_equity`.
- Il retraining aggiunge diagnostica out-of-sample deterministica (`oos_sharpe`, `oos_max_drawdown`, `oos_turnover`, `walk_forward_window_count`, `pbo`).
- Il promotion gate blocca candidati senza diagnostica walk-forward sufficiente, con `oos_sharpe <= 0` o con `pbo > 0.50`.
- Le convenzioni operative di Fase 2 sono riepilogate in [`docs/fase2-realism.md`](./docs/fase2-realism.md).

## Fase 3 Operational Controls

- `api/services/trading_service.py` ora esegue un preflight unico con paper-mode enforcement, kill switch operativo, pre-trade risk e reconciliation.
- Gli ordini generati dalla pipeline vengono normalizzati da notional USD a share quantity usando i prezzi correnti prima dell'invio ad Alpaca.
- `POST /api/trading/execute` blocca l'esecuzione con HTTP `409` quando il preflight rileva un hard stop.
- Nuovi endpoint: `GET /api/trading/preflight/{date}` e `GET /api/trading/reconcile/{date}`.
- I run di paper trading persistono artifact operativi in `data/operations/{date}.json` e report di rischio in `data/risk/risk_report_{date}.json`.
- Il monitoring settings service espone anche `MLCOUNCIL_AUTOMATION_PAUSED`, `MLCOUNCIL_MAX_DAILY_ORDERS`, `MLCOUNCIL_MAX_TURNOVER` e `MLCOUNCIL_MAX_POSITION_SIZE`.
- Le convenzioni operative di Fase 3 sono riepilogate in [`docs/fase3-operational-controls.md`](./docs/fase3-operational-controls.md).

### Run the Pipeline (Demo)

```bash
python scripts/run_pipeline.py
```

This runs a standalone demo pipeline that:
1. Downloads OHLCV data
2. Computes Alpha158 features
3. Trains LightGBM and HMM models
4. Generates signals and portfolio weights
5. Outputs daily orders

### Run Dagster Pipeline (Production)

```bash
# Start Dagster UI
dagster dev -f data/pipeline.py

# Or run programmatically
python -c "from data.pipeline import defs; defs.get_job('daily_pipeline').execute_in_process()"
```

### Run Public Dashboard

```bash
streamlit run dashboard/app.py
```

Access at: http://localhost:8501

### Run Admin Dashboard

```bash
python run_admin.py
```

Access at: http://localhost:8000

- **Admin UI**: http://localhost:8000
- **API Docs**: http://localhost:8000/api/docs

## API Endpoints

### Health
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | System health status |
| GET | `/api/health/dagster` | Dagster connectivity |

### Pipeline
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/pipeline/run` | Trigger pipeline execution |
| GET | `/api/pipeline/status` | Last run status |

### Portfolio
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/portfolio/weights` | Current portfolio weights |
| GET | `/api/portfolio/orders/dates` | Available order dates |
| GET | `/api/portfolio/orders/{date}` | Orders for specific date |

### Configuration
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/config/universe` | Get universe config |
| PUT | `/api/config/universe` | Update universe config |
| GET | `/api/config/models` | Get model hyperparameters |
| GET | `/api/config/regime-weights` | Get regime weights |
| PUT | `/api/config/regime-weights` | Update regime weights |

### Monitoring
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/monitoring/alerts` | Current active alerts |
| GET | `/api/monitoring/alerts/history` | Alert history |

### Trading
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/trading/status` | Alpaca paper status + runtime flags |
| GET | `/api/trading/orders/latest` | Latest order date |
| GET | `/api/trading/orders/pending/{date}` | Pending orders for date |
| GET | `/api/trading/preflight/{date}` | Pre-trade risk and reconciliation snapshot |
| GET | `/api/trading/reconcile/{date}` | Target vs current positions summary |
| POST | `/api/trading/execute` | Execute paper orders, blocked with `409` on hard stops |
| POST | `/api/trading/liquidate` | Liquidate all current paper positions |
| GET | `/api/trading/history` | Local trade log history |

## Configuration

### Universe (`config/universe.yaml`)

```yaml
universe:
  tickers: [AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA, JPM, JNJ, V, UNH, XOM, WMT, PG, MA]
settings:
  data_dir: data/raw
  transaction_timezone: America/New_York
  transaction_time_hour: 20
  transaction_time_minute: 30
macro:
  fred_series:
    vix: VIXCLS
    treasury_10y: DGS10
    treasury_2y: DGS2
    sp500: SP500
```

### Model Parameters (`config/models.yaml`)

```yaml
lgbm:
  n_estimators: 500
  learning_rate: 0.05
  num_leaves: 64
  
cpcv:
  n_splits: 6
  embargo_days: 5
  n_test_folds: 2

hmm:
  n_states: 3
  n_iter: 100

sentiment:
  model_name: "ProsusAI/finbert"
  recency_decay: 0.7
```

### Regime Weights (`config/regime_weights.yaml`)

```yaml
regime_weights:
  bull:       {lgbm: 0.50, sentiment: 0.30, hmm: 0.20}
  bear:       {lgbm: 0.40, sentiment: 0.20, hmm: 0.40}
  transition: {lgbm: 0.45, sentiment: 0.25, hmm: 0.30}

weight_clip:
  min: 0.05
  max: 0.70
```

## Testing

```bash
# Run all API tests
python -m pytest tests/test_api_*.py -v

# Run specific test modules
python -m pytest tests/test_council.py -v
python -m pytest tests/test_models.py -v
```

## Project Structure

```
MLCouncil/
├── api/                    # FastAPI Admin Backend
│   ├── main.py            # App factory
│   ├── routers/           # API endpoints
│   ├── services/          # Business logic
│   ├── static/            # CSS/JS for admin UI
│   └── templates/         # HTML templates
├── backtest/              # NautilusTrader backtest engine
├── council/               # Ensemble aggregation & portfolio
│   ├── aggregator.py      # CouncilAggregator
│   ├── portfolio.py       # PortfolioConstructor
│   ├── conformal.py       # ConformalPositionSizer
│   ├── monitor.py         # CouncilMonitor
│   └── alerts.py          # AlertDispatcher
├── config/                # YAML configuration files
├── dashboard/             # Streamlit public dashboard
├── data/                  # Data pipeline & storage
│   ├── ingest/           # Market data, news, macro
│   ├── features/         # Alpha158, targets
│   ├── store/            # ArcticDB feature store
│   └── pipeline.py       # Dagster assets & jobs
├── models/                # ML models
│   ├── technical.py      # LightGBM model
│   ├── sentiment.py      # FinBERT model
│   └── regime.py         # HMM model
├── scripts/               # Utility scripts
├── tests/                 # Test suite
├── requirements.txt       # Main dependencies
├── requirements_api.txt   # API-only dependencies
└── run_admin.py          # Admin server entry point
```

## Deployment

### Docker (Recommended)

Build and run with Docker Compose:

```bash
# Build all services
docker-compose build

# Run admin API only
docker-compose up admin-api

# Run all services (API, Dashboard, Dagster, MLflow)
docker-compose up

# Run in background
docker-compose up -d
```

**Services:**
| Service | Port | Description |
|---------|------|-------------|
| admin-api | 8000 | FastAPI Admin backend |
| dashboard | 8501 | Streamlit public dashboard |
| dagster | 3000 | Pipeline orchestration UI |
| mlflow | 5000 | Experiment tracking |

**Production deployment:**
```bash
# Build image
docker build -t mlcouncil .

# Run API
docker run -p 8000:8000 -v ./data:/app/data mlcouncil

# Run Dashboard
docker run -p 8501:8501 mlcouncil streamlit run dashboard/app.py
```

### Streamlit Cloud (Dashboard Only)

1. Push to GitHub
2. Connect to Streamlit Cloud
3. Set `dashboard/app.py` as entry point
4. Configure secrets for API keys

### Render/Railway (API)

1. Connect GitHub repository
2. Set build command: `pip install -r requirements.txt`
3. Set start command: `python run_admin.py`
4. Configure environment variables

## Roadmap

- [ ] Live trading integration with Alpaca
- [ ] Additional alpha models (momentum, fundamental)
- [ ] Multi-asset support (crypto, futures)
- [ ] Real-time dashboard updates via WebSockets
- [ ] User authentication for admin panel
- [ ] Automated model retraining pipeline

## License

MIT License - See LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Acknowledgments

- [NautilusTrader](https://nautilustrader.io/) - Backtesting engine
- [ArcticDB](https://github.com/man-group/ArcticDB) - Feature store
- [Evidently](https://evidentlyai.com/) - Drift detection
- [Dagster](https://dagster.io/) - Pipeline orchestration
