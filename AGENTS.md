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
# Unified UI: served by the admin API at http://localhost:8000 (login: MLCOUNCIL_ADMIN_USERNAME/PASSWORD from .env)
dagster dev -f data/pipeline.py     # Pipeline UI :3000

# Docker
docker compose up -d
docker compose -f docker-compose.yml -f docker-compose.observability.yml --profile observability up -d
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

## TFT alpha challenger (T2.1 — shadow only)

Temporal Fusion Transformer challenger is **not** in the daily Dagster path.

```bash
python scripts/train_tft.py --start 2021-01-01 --end 2024-12-31
python scripts/run_walkforward_promotion.py --model tft --dry-run
python -m pytest tests/test_tft.py -v
```

- `models/tft.py` — `TemporalFusionAlpha` (PyTorch VSN+GRU+attention; CPU inference SLO <300ms on fixture)
- Shadow outputs: `data/results/tft_shadow_signals.parquet`, `data/results/walkforward_signals_tft.parquet`
- Promotion compares TFT vs LightGBM champion via T1.1 gate (`torch>=2.0` in requirements)
- ADR: `docs/adr/2026-05-21-tft-alpha-challenger.md`

## Online learning (T1.2)

Daily incremental LightGBM refit is **off by default**; **active via canary since gate G1 (2026-08-13)** when the canary controller applies `MLCOUNCIL_ONLINE_LEARNING=true` (`config/canary.yaml`). Enable for staging/paper:

```bash
export MLCOUNCIL_ONLINE_LEARNING=true
# optional: MLCOUNCIL_ONLINE_IC_THRESHOLD=0.05  MLCOUNCIL_ONLINE_REFIT_DAYS=60
```

- `models/online.py` — `IncrementalLightGBM`, IC gate, `run_daily_incremental_update()`
- `council/drift.py` — ADWIN on 60d equal-weight returns; DDM on binary error indicators
- Dagster `lgbm_signals` refits champion before predict when enabled; walk-forward CI still owns promotion
- ADR: `docs/adr/2026-05-21-online-learning.md`

## Canary activation (F-0.4 — G1 approved 2026-08-13)

Shadow features activate via `config/canary.yaml` + `council/canary.py` (`CanaryController`:
run-policy env injection `apply()` — operator env wins; `record()`/`check_revert()` sticky
revert when the configured metric stays below `floor` for `min_days` consecutive runs; alerts
via `council/alerting.py`). Daily asset `canary_health` records same-day council metrics;
no-op with zero side effects when no feature is enabled.

G1-approved active trio: online learning (`MLCOUNCIL_ONLINE_LEARNING=true`), CQR sizing
(`MLCOUNCIL_POSITION_SIZING=cqr`), dynamic slippage (`MLCOUNCIL_DYNAMIC_SLIPPAGE=true`).
`moe_gating` is **NOT** active (gate untrained — train hard-EM gating first, then canary).
Reverts land in `data/results/canary_state.json` and dispatch CRITICAL alerts through the
standard channels. Flag inventory with expiry dates: `docs/flag-registry-2026-08-13.md`.
ADR: `docs/adr/2026-08-13-canary-config-profile.md`.

## Production profile (gated — default)

```bash
cp .env.example .env
# Set ALPACA_*, POLYGON_*, MLCOUNCIL_API_KEY, ALERT_EMAIL, SMTP_PASSWORD
python scripts/setup_prod.py
dagster dev -f data/pipeline.py
```

- `MLCOUNCIL_ENV_PROFILE=prod` + `MLCOUNCIL_USE_PRODUCTION_MANIFEST=true`
- Champions: `config/production_manifest.yaml` (LightGBM, FinBERT, HMM, linear/conformal/Ledoit/CVXPY)
- Weekly: `model_promotion_gate` Dagster asset + GitHub `walk-forward-ci.yml`
- Seed CI caches: `python scripts/populate_walkforward_caches.py`
- Promote after 3 passes: `python scripts/promote_model.py --model lightgbm`
- Council module promote: `python scripts/promote_council_module.py --module dcc`
- Staging TFT path: `python scripts/establish_wave2_staging_promotion.py --model tft`

ADR: `docs/adr/2026-05-21-production-promotion-gate.md`

## Frontier profile (R&D only — no gate)

```bash
MLCOUNCIL_ENV_PROFILE=frontier
python scripts/bootstrap_frontier.py
```

Bypasses promotion gate; not for paper/live without explicit acceptance of risk.

## Wave 3 council/portfolio (T3.x — off by default)

Production paths unchanged unless env flags are set. Train checkpoints before enabling shadow modes:

```bash
python scripts/train_moe_gating.py              # → models/checkpoints/moe_gate.pkl
python scripts/train_stacking_cqr.py --cqr --stacking
python scripts/train_alpha_portfolio_end2end.py # E2E scaffold summary

# MoE gating (default linear)
export MLCOUNCIL_AGGREGATOR_MODE=moe

# CQR sizing (default conformal / MAPIE); pipeline loads cqr_sizer.pkl or conformal_sizer.pkl
export MLCOUNCIL_POSITION_SIZING=cqr
export MLCOUNCIL_STACKING_SHADOW=true           # logs data/results/shadow_stacking/
export MLCOUNCIL_STACKING_BACKEND=ridge         # or xgb when installed

# DCC-GARCH covariance (default ledoit in pipeline)
export MLCOUNCIL_COVARIANCE_ESTIMATOR=dcc

# Differentiable portfolio (default cvxpy; uses get_portfolio_constructor())
export MLCOUNCIL_PORTFOLIO_MODE=diff
```

- Factories: `get_position_sizer()`, `get_portfolio_constructor()`, `compute_covariance_from_returns()`
- Modules: `council/moe_gating.py`, `council/cqr.py`, `council/covariance_dynamic.py`, `council/portfolio_diff.py`
- Pipeline: `council_signal` (stacking shadow), `portfolio_weights` (sizer + portfolio factories)
- ADRs: `docs/adr/2026-05-21-moe-gating.md`, `stacking-cqr.md`, `dynamic-covariance.md`, `differentiable-portfolio.md`
- Tests: `tests/test_moe_gating.py`, `test_cqr.py`, `test_dcc_garch.py`, `test_portfolio_diff.py`

## Observability

OpenTelemetry tracing is **off by default**. Enable for local/debug runs:

```bash
# Host processes → collector on localhost
docker compose -f docker-compose.yml -f docker-compose.observability.yml --profile observability up -d
export MLCOUNCIL_OTEL_ENABLED=true
export OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4318/v1/traces
python scripts/run_pipeline.py --partition 2026-05-20

# All services in compose (default in docker-compose.yml):
# MLCOUNCIL_OTEL_ENABLED=true
# OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4318/v1/traces
```

- `observability/tracing.py` — `init_tracing()`, `trace_span()` (no-op when disabled)
- Dagster spans on `raw_ohlcv`, `alpha158_features`, `lgbm_signals`, `daily_orders`
- Dashboard: `dashboards/grafana/mlcouncil.json` (provisioned by observability compose)
- ADR: `docs/adr/2026-05-21-otel-grafana.md`

## Wave 4 execution & risk (T4.x — shadow scaffolds)

- `council/causal_drift.py`, `council/tda_warning.py`, `council/generative_stress.py`
- `execution/rl_agent.py`, `execution/router.py`, `execution/lob_simulator.py`
- Dagster: `tda_warning_signal` (weekly); monitor: `check_causal_graph_drift`
- RiskEngine: `compute_var(..., method="generative")`
- ADRs: `docs/adr/2026-05-21-causal-drift-pcmci.md`, `tda-early-warning.md`, `generative-stress.md`, `rl-execution.md`, `smart-order-routing.md`
- UI: Challenger Promotion page in the unified SPA (`frontend/src/pages/PromotionPage.tsx`)

## Known Issues

- `scripts/run_pipeline.py:252` — Sentiment now downloads real news via Yahoo Finance RSS (use `--with-sentiment` to enable)
- `api/services/analytics_service.py:646` — Drawdown delta now calculates day-over-day change
- `docs/architecture-as-is-to-be-2026-05-21.md` tracks current AS IS/TO BE drift from the combined codebase analysis; consult it before starting large quant, risk, or dashboard work.
