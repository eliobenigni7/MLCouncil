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
- `council/aggregator.py` — Combines active alpha signals with regime-conditional weights; current daily path passes LightGBM + sentiment while HMM supplies the regime label
- `council/portfolio.py` — CVXPY optimizer, outputs `data/orders/{date}.parquet`
- `data/store/arctic_store.py` — Feature store with point-in-time versioning

## Key Quirks

**Test stub.** `tests/conftest.py` installs a `slowapi` stub so tests run without the package installed. Don't add slowapi to test requirements.

**Pipeline comments are Italian.** `data/pipeline.py` uses Italian comments. Preserve this style.

**Universe config.** `config/universe.yaml` supports two formats: legacy `universe.tickers` list or bucketed (`large_cap`, `mid_cap`). Pipeline loader handles both.

**Current universe shape.** Runtime config is bucketed equities (`universe.large_cap`, `universe.mid_cap`) plus `crypto_universe.large_cap` (`BTCUSD`, `ETHUSD`), with crypto support still in progress operationally.

**Technical feature lookahead prevention.** The Alpha158-inspired feature set is shifted 1 day to avoid lookahead bias in backtesting. Don't change this without understanding the implications.

**Council performance weighting wording.** Use "EWM IC-Sharpe over recent history (halflife up to 20, bounded by configured history window)" instead of "rolling 100-day IR".

**Daily inference scope.** Daily Dagster inference does not compute training targets; `compute_targets` belongs to training/backtest flows.

**Transaction costs wording.** Current transaction costs are configurable heuristics (runtime defaults from env/code), not a realized-slippage calibrated impact model.

**No Python formatter/linter configured.** Follow existing code style when editing.

## Requirements Structure

- `requirements.txt` — Core + includes `-r requirements_api.txt`
- `requirements_api.txt` — FastAPI, Alpaca, ArcticDB
- `requirements_dashboard.txt` — Streamlit-only (for cloud deploy)

## Test Gaps

Historically weak areas now have tests, but keep coverage tight when editing:
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
- `docs/architecture-as-is-to-be-2026-05-21.md` tracks current AS IS/TO BE drift from the combined codebase analysis; consult it before starting large quant, risk, or dashboard work.

## Cursor Cloud specific instructions

### Environment

- Python 3.12 venv at `.venv/`. Activate with `source .venv/bin/activate`.
- Dependencies: `pip install -r requirements.txt -r requirements_ci.txt` (includes all runtime + linting/testing deps).
- A `.env` file in the repo root with `MLCOUNCIL_ENV_PROFILE=local` is sufficient to start services without external API keys.

### Running services

- **FastAPI Admin API** (`python run_admin.py`, port 8000): starts immediately, no external dependencies needed for local profile.
- **Streamlit Dashboard** (`streamlit run dashboard/app.py --server.headless true`, port 8501): reads data from local files; works without API keys.
- **Dagster** (`dagster dev -f data/pipeline.py`, port 3000): optional, only needed for pipeline orchestration UI.
- Pipeline API endpoint (`POST /api/pipeline/run`) requires the Dagster server to be running; standalone use: `python scripts/run_pipeline.py`.

### Testing caveats

- `tests/test_models.py` includes LightGBM CPCV training fixtures (`fitted_lgbm`) that take several minutes on 4-CPU VMs. Skip with `--ignore=tests/test_models.py` for fast iteration; include for full coverage runs.
- `tests/test_retraining.py` also has slow model-fitting tests; can be skipped similarly.
- There are 8 pre-existing test failures (config-consistency assertions that depend on env state); these are not regressions.
- Run fast tests: `python -m pytest tests/ --ignore=tests/test_models.py --ignore=tests/test_retraining.py`

### Linting

- `ruff check` (see README "Testing" section for target file list). Pre-existing E402/F841 warnings.
- `mypy --config-file mypy.ini` — configured with `follow_imports = skip`.
- `bandit -q -r api council execution runtime_env.py -lll` — clean.

### No external services required for local dev

PostgreSQL, MinIO, MLflow server, and Alpaca API are all optional for the `local` profile. The system uses LMDB/Parquet storage and gracefully degrades when external services are absent.
