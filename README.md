# MLCouncil

MLCouncil is an end-to-end paper trading platform for US equities.
It ingests market, news, and macro data, builds features, trains and evaluates multiple alpha models, aggregates them with regime-aware weights, optimizes a target portfolio, and can automatically submit paper orders to Alpaca after a successful pipeline run.

## How It Works

At a high level, the system runs this loop:

1. Ingest daily OHLCV, news, and macro data.
2. Build point-in-time features with lookahead protection.
3. Train or refresh the alpha models and log runs to MLflow.
4. Generate model signals and combine them in the council.
5. Optimize target weights under trading and risk constraints.
6. Write dated orders with lineage metadata.
7. Run pre-trade checks, reconciliation, and risk validation.
8. Submit paper orders to Alpaca automatically or manually.
9. Expose status, positions, fills, monitoring, and diagnostics in the Admin UI, Dashboard, Dagster, and MLflow.

In production-style local usage, Dagster orchestrates the pipeline, the FastAPI admin service exposes operations and controls, Streamlit exposes the read-only dashboard, and MLflow tracks experiments and backtests.

## Architecture

```text
                                  +----------------------+
                                  |      MLflow          |
                                  | runs, metrics, tags  |
                                  +----------+-----------+
                                             ^
                                             |
+-------------+      +-------------+      +--+----------------+      +-------------------+
| data/ingest | ---> | data/features| ---> | models + council | ---> | council/portfolio |
| OHLCV/news  |      | Alpha158 etc |      | signals, weights |      | target weights    |
+------+------+      +------+------+      +---------+---------+      +---------+---------+
       |                    |                        |                          |
       v                    v                        v                          v
+-------------+      +-------------+      +-------------------+      +-------------------+
| raw datasets |      | feature sets |      | model artifacts   |      | data/orders/*.parquet |
+-------------+      +------+------+      +-------------------+      +---------+---------+
                             |                                                  |
                             v                                                  v
                       +-------------+                                +-------------------+
                       | ArcticDB /  |                                | trading_service   |
                       | local store |                                | preflight, risk,  |
                       +------+------+                                | reconcile, submit |
                              |                                       +---------+---------+
                              |                                                 |
                              v                                                 v
                       +-------------+                                +-------------------+
                       |  Dagster    |                                | Alpaca Paper      |
                       | orchestration|                               | orders and fills  |
                       +------+------+                                +---------+---------+
                              |                                                 |
                              +---------------------+---------------------------+
                                                    |
                         +--------------------------+--------------------------+
                         |                                                     |
                         v                                                     v
                +-------------------+                                 +-------------------+
                | FastAPI Admin UI  |                                 | Streamlit Dashboard |
                | control plane     |                                 | monitoring/read-only|
                +-------------------+                                 +-------------------+
```

## Core Components

### Pipeline and Data Layer
- `data/pipeline.py`: Dagster assets, jobs, partitions, and asset checks.
- `data/ingest/`: market, news, and macro ingestion.
- `data/features/`: Alpha158, sentiment, sector exposure, and related feature builders.
- `data/store/arctic_store.py`: point-in-time feature storage.

### Models and Council
- `models/technical.py`: LightGBM-based technical alpha model.
- `models/sentiment.py`: sentiment alpha model.
- `models/regime.py`: regime detection model.
- `council/aggregator.py`: combines model outputs with regime-conditional weights.
- `council/portfolio.py`: converts council signals into feasible portfolio weights.

### Trading and Operations
- `api/services/trading_service.py`: preflight checks, reconciliation, execution, live status, trade history.
- `api/services/pipeline_automation.py`: monitors Dagster runs and auto-executes orders when enabled.
- `execution/alpaca_adapter.py`: Alpaca paper broker adapter.
- `council/risk_engine.py`: operational risk checks used before execution.

### Interfaces
- `api/`: FastAPI backend and Admin UI.
- `dashboard/`: Streamlit dashboard.
- `docker-compose.yml`: local multi-service runtime.

## Daily Operational Flow

### 1. Pipeline Run
A Dagster run produces the daily datasets and artifacts:
- raw market, news, macro inputs
- feature tables
- model outputs and council scores
- optimized portfolio weights
- `data/orders/{date}.parquet`

### 2. Lineage and Contracts
Orders and model-related artifacts carry minimum lineage metadata:
- `pipeline_run_id`
- `data_version`
- `feature_version`
- `model_version`

Dagster asset checks fail closed when the expected contracts are violated.

### 3. Pre-Trade Controls
Before orders are sent, the trading service evaluates:
- paper-mode enforcement
- kill switch / automation pause
- max daily order count
- max turnover
- max position size
- projected risk breaches
- reconciliation between current positions and target orders

### 4. Execution
If preflight is green, the service:
- normalizes notionals to share quantities
- submits paper orders to Alpaca
- records operations artifacts and trade logs
- updates trade history with live broker order status

### 5. Monitoring
The system exposes runtime status through:
- Admin UI: operational control and execution status
- Dashboard: portfolio, regime, attribution, alerts
- Dagster UI: orchestration state
- MLflow UI: training, retraining, backtest, promotion evidence

## Main Features

### Ensemble Alpha Stack
- Technical model with Alpha158-style features
- Sentiment model built from financial news
- Regime-aware weighting logic in the council
- Adaptive aggregation based on model diagnostics and regime context

### Portfolio Construction
- Long-only optimization
- Max position cap
- Turnover control
- Sector exposure handling
- Feasible bootstrap behavior for an empty paper portfolio

### Paper Trading Controls
- Automatic order execution after successful Dagster runs
- Preflight and reconciliation endpoints
- Kill switch support
- Runtime environment validation
- Persistent operations and risk artifacts

### Experiment Tracking and Validation
- Standardized MLflow runs and tags
- Backtest metrics with gross/net realism
- Walk-forward diagnostics and promotion gates
- Data, feature, and model lineage propagation

## Quick Start

### Local Python Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Required Environment

Create a `.env` file in the project root:

```env
ALPACA_API_KEY=your_alpaca_key
ALPACA_SECRET_KEY=your_alpaca_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets
ARCTICDB_URI=lmdb://data/arctic/
MLFLOW_TRACKING_URI=http://localhost:5000
```

Runtime profiles and examples live under `config/`:
- `config/runtime.env`
- `config/runtime.local.env.example`
- `config/runtime.paper.env.example`

## Run the System

### Docker Compose

```bash
docker compose build
docker compose up -d
```

Services:
- Admin UI and API: `http://localhost:8000`
- Streamlit Dashboard: `http://localhost:8501`
- Dagster UI: `http://localhost:3000`
- MLflow UI: `http://localhost:5000`

### Local Services Without Docker

```bash
python run_admin.py
streamlit run dashboard/app.py
dagster dev -f data/pipeline.py
```

## Pipeline and Trading Usage

### Trigger a Pipeline Run

```bash
curl -X POST http://localhost:8000/api/pipeline/run \
  -H "Content-Type: application/json" \
  -d '{"partition":"2026-04-07"}'
```

### Check Automation Status

```bash
curl http://localhost:8000/api/pipeline/automation/<run_id>
```

### Manual Preflight and Execution

```bash
curl http://localhost:8000/api/trading/preflight/2026-04-07
curl -X POST http://localhost:8000/api/trading/execute \
  -H "Content-Type: application/json" \
  -d '{"date":"2026-04-07"}'
```

### Enable Auto-Execute

Set:

```env
MLCOUNCIL_AUTO_EXECUTE=true
```

When enabled, a successful pipeline run automatically monitors the Dagster run and executes the matching order date through the trading service.

## Important Endpoints

### Health and Runtime
- `GET /api/health`
- `GET /api/health/dagster`
- `GET /api/trading/status`

### Pipeline
- `POST /api/pipeline/run`
- `GET /api/pipeline/status`
- `GET /api/pipeline/automation/{run_id}`

### Trading
- `GET /api/trading/orders/latest`
- `GET /api/trading/orders/pending/{date}`
- `GET /api/trading/preflight/{date}`
- `GET /api/trading/reconcile/{date}`
- `POST /api/trading/execute`
- `POST /api/trading/liquidate`
- `GET /api/trading/history`

### Configuration and Monitoring
- `GET /api/config/universe`
- `PUT /api/config/universe`
- `GET /api/config/regime-weights`
- `PUT /api/config/regime-weights`
- `GET /api/monitoring/alerts`
- `GET /api/monitoring/alerts/history`

## Project Structure

```text
MLCouncil/
|-- api/                  FastAPI backend, Admin UI, service layer
|-- backtest/             Backtest engine and validation
|-- config/               Runtime and model configuration
|-- council/              Aggregation, portfolio, risk, monitoring
|-- dashboard/            Streamlit dashboard
|-- data/                 Ingestion, features, storage, Dagster pipeline
|-- docs/                 Operational documentation and phase notes
|-- execution/            Broker adapter(s)
|-- models/               Alpha and regime models
|-- scripts/              Utility and support scripts
|-- tests/                Pytest suite
|-- docker-compose.yml    Local multi-service stack
|-- requirements.txt      Main dependencies
`-- run_admin.py          Admin server entry point
```

## Documentation

### Foundation and Architecture Phases
- [docs/fase1-foundations.md](docs/fase1-foundations.md)
- [docs/fase2-realism.md](docs/fase2-realism.md)
- [docs/fase3-operational-controls.md](docs/fase3-operational-controls.md)
- [docs/fase4-hardening.md](docs/fase4-hardening.md)

### Operations
- [docs/paper-trading-runbook.md](docs/paper-trading-runbook.md)
- [docs/model-promotion-criteria.md](docs/model-promotion-criteria.md)

## Testing

```bash
python -m pytest
python -m pytest tests/test_council.py -v
python -m pytest tests/test_trading_service.py -v
```

## Current Scope

The current target is robust paper trading on US equities with Alpaca Paper.
Kubernetes, GitOps, and live trading are intentionally out of scope until the paper-trading path remains stable end to end.

## License

MIT License. See `LICENSE`.
