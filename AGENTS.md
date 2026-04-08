# AGENTS.md

Compact guidance for OpenCode sessions working in this repository.

## Commands

```bash
# Setup
python -m venv .venv
.venv\Scripts\activate              # Windows
pip install -r requirements.txt    # includes requirements_api.txt

# Tests
python -m pytest                    # all tests
python -m pytest tests/test_council.py -v
python -m pytest tests/ -k "test_aggregator"

# Services
python run_admin.py                 # FastAPI :8000
streamlit run dashboard/app.py      # Dashboard :8501
dagster dev -f data/pipeline.py     # Pipeline UI :3000
```

No lint or typecheck commands configured in this repo.

## Architecture

```
data/ingest → data/features → models/ → council/ → execution/
                              ↓
                        ArcticDB (LMDB)
```

- `data/pipeline.py` — Dagster orchestration (4 layers: ingest→features→signals→council)
- `council/aggregator.py` — Combines 3 alpha models with regime-conditional weights
- `council/portfolio.py` — CVXPY optimizer, outputs `data/orders/{date}.parquet`
- `data/store/arctic_store.py` — Feature store with point-in-time versioning

## Key Quirks

**Test stub.** `tests/conftest.py` installs a `slowapi` stub so tests run without the package installed. Don't add slowapi to test requirements.

**Pipeline comments are Italian.** `data/pipeline.py` uses Italian comments. Preserve this style.

**Universe config.** `config/universe.yaml` supports two formats: legacy `universe.tickers` list or bucketed (`large_cap`, `mid_cap`). Pipeline loader handles both.

**Alpha158 lookahead prevention.** Features are shifted 1 day to avoid lookahead bias in backtesting. Don't change this without understanding the implications.

**No Python formatter/linter configured.** Follow existing code style when editing.

## Requirements Structure

- `requirements.txt` — Core + includes `-r requirements_api.txt`
- `requirements_api.txt` — FastAPI, Alpaca, ArcticDB
- `requirements_dashboard.txt` — Streamlit-only (for cloud deploy)

## Test Gaps

These modules lack tests:
- `data/store/arctic_store.py`
- `execution/alpaca_adapter.py`
- `council/risk_engine.py`

## Paper Trading

Trading service at `api/services/trading_service.py` executes pipeline orders to Alpaca Paper:
- UI: Admin page "Trading" (http://localhost:8000) — status, positions, pending orders, execute button
- API: `POST /api/trading/execute` with `{"date": "YYYY-MM-DD"}`
- Safety: max 20 orders/day, 30% turnover, 10% position cap (configurable in `config/runtime.env`)

Required env vars (in `.env`):
```
ALPACA_API_KEY=...
ALPACA_SECRET_KEY=...
ALPACA_BASE_URL=https://paper-api.alpaca.markets
MLCOUNCIL_MAX_DAILY_ORDERS=20
MLCOUNCIL_MAX_TURNOVER=0.30
MLCOUNCIL_MAX_POSITION_SIZE=0.10
```

## Known Issues

- `scripts/run_pipeline.py:252` — Sentiment now downloads real news via Yahoo Finance RSS (use `--with-sentiment` to enable)
- `dashboard/data_loader.py:567` — Drawdown delta now calculates day-over-day change