# Flag Registry — `MLCOUNCIL_*` configuration inventory

- Date: 2026-08-13
- Roadmap: F-0.4 "Flag governance + canary layer" (`docs/roadmap-2026-2030-autonomous-council.md`)
- Scope: every `MLCOUNCIL_*` env flag read by the codebase (excluding `.venv/`, `data/`, `.claude/`, `.worktrees/`)

This registry implements F-0.4: **each disabled-by-default feature carries a
docstring annotation, an expiry date, and a target phase**. The expiry date is
the *decision date* (promote via canary or retire), not a deletion date.

Design references: canary controller `council/canary.py` and activation config
`config/canary.yaml` (G1 owner gate, automatic revert on metric regression).
Flags listed there are marked "canary-controlled (G1)".

## Legend

- **Status** — `disabled-by-default` (shadow/experimental, off unless set), `canary-controlled (G1)` (activated via `config/canary.yaml` after owner approval), `active/ops` (production path, no expiry), `param` (tuning knob of a parent feature), `doc-only` (referenced but no code read).
- **Target phase / Expiry** — roadmap phase that decides promote-or-retire, and the decision date. `—` for active/ops flags.

## P-1.1 — canary candidates (expiry **2027-02-01**)

| Flag | Default | Module | Purpose | Status | Target phase | Expiry |
|---|---|---|---|---|---|---|
| `MLCOUNCIL_ONLINE_LEARNING` | `false` | `models/online.py` | Incremental daily LightGBM refit of the champion (IC-gated) | canary-controlled (G1: `online_learning`) | P-1.1 | 2027-02-01 |
| `MLCOUNCIL_AGGREGATOR_MODE` | `linear` | `council/moe_gating.py`, `council/aggregator.py` | Council aggregation mode (`linear` default, `moe` experimental) | canary-controlled (G1: `moe_gating`) | P-1.1 | 2027-02-01 |
| `MLCOUNCIL_POSITION_SIZING` | `conformal` | `council/cqr.py` | Position sizer factory (`conformal` default, `cqr` experimental) | canary-controlled (G1: `position_sizing_cqr`) | P-1.1 | 2027-02-01 |
| `MLCOUNCIL_COVARIANCE_ESTIMATOR` | `ledoit` | `council/covariance_dynamic.py` | Covariance estimator (`ledoit` default, `dcc`, `factor`) | disabled-by-default | P-1.1 | 2027-02-01 |
| `MLCOUNCIL_STACKING_SHADOW` | `false` | `council/cqr.py` | Log stacking meta-learner shadow predictions | disabled-by-default | P-1.1 | 2027-02-01 |
| `MLCOUNCIL_STACKING_BACKEND` | `ridge` | `council/cqr.py` | Stacking backend (`ridge` default, `xgb` when installed) | param | P-1.1 | 2027-02-01 |
| `MLCOUNCIL_USE_STACKED_COUNCIL` | `false` | `council/frontier.py` | Use stacked council signals in the daily aggregator | disabled-by-default | P-1.1 | 2027-02-01 |
| `MLCOUNCIL_META_LABEL` | `false` | `models/meta_label.py` | Meta-label gate: zero signals below threshold | disabled-by-default | P-1.1 | 2027-02-01 |
| `MLCOUNCIL_META_LABEL_SHADOW` | `false` | `models/meta_label.py` | Meta-label shadow (log only) | disabled-by-default | P-1.1 | 2027-02-01 |
| `MLCOUNCIL_META_LABEL_THRESHOLD` | `0.55` | `models/meta_label.py` | Meta-label probability cutoff | param | P-1.1 | 2027-02-01 |
| `MLCOUNCIL_REGIME_DSS_SHADOW` | `false` | `models/regime_dss.py` | Deep state-space regime challenger (T2.3) shadow | disabled-by-default | P-1.1 | 2027-02-01 |
| `MLCOUNCIL_MULTI_PERIOD_TC` | `false` | `council/portfolio_multiperiod.py` | Multi-period target smoothing under transaction costs | disabled-by-default | P-1.1 | 2027-02-01 |
| `MLCOUNCIL_MP_HORIZON_DAYS` | `5` | `council/portfolio_multiperiod.py` | Multi-period horizon H | param | P-1.1 | 2027-02-01 |
| `MLCOUNCIL_MP_RISK_AVERSION` | `1.0` | `council/portfolio_multiperiod.py` | Multi-period risk aversion | param | P-1.1 | 2027-02-01 |
| `MLCOUNCIL_MP_TC_LAMBDA` | `2.0` | `council/portfolio_multiperiod.py` | Multi-period TC penalty weight | param | P-1.1 | 2027-02-01 |
| `MLCOUNCIL_MP_SMOOTHING` | `0.5` | `council/portfolio_multiperiod.py` | Target smoothing factor | param | P-1.1 | 2027-02-01 |
| `MLCOUNCIL_HRP_SOFT_PRIOR` | `false` | `council/hrp.py`, `council/portfolio.py` | Blend HRP weights as soft prior on CVXPY solution | disabled-by-default | P-1.1 | 2027-02-01 |
| `MLCOUNCIL_HRP_BLEND` | `0.25` | `council/portfolio.py`, `council/hrp.py` | HRP blend weight (`hrp_blend` mode default 0.5) | param | P-1.1 | 2027-02-01 |
| `MLCOUNCIL_HRP_BLEND_WEIGHTING` | `fixed` | `council/hrp.py` | HRP blend weighting scheme (`fixed`/IR) | param | P-1.1 | 2027-02-01 |
| `MLCOUNCIL_HRP_IR_COND_REF` | `100` | `council/hrp.py` | Reference window for IR-conditional HRP blend | param | P-1.1 | 2027-02-01 |
| `MLCOUNCIL_TFT_IN_COUNCIL` | `false` | `council/frontier.py`, `models/tft.py` | Wire TFT signals into the daily council | disabled-by-default | P-1.1 | 2027-02-01 |
| `MLCOUNCIL_PORTFOLIO_MODE` | `cvxpy` | `council/portfolio_diff.py` | Portfolio constructor (`cvxpy` default, `diff`, `hrp_blend`) | disabled-by-default | P-1.1 | 2027-02-01 |
| `MLCOUNCIL_LLM_SENTIMENT_SHADOW` | `false` | `models/sentiment_llm.py` | FinMA/FinGPT LLM sentiment challenger (T2.2) shadow | disabled-by-default | P-1.1 | 2027-02-01 |
| `MLCOUNCIL_LLM_SENTIMENT_MOCK` | `false` | `models/sentiment_llm.py` | Mock LLM responses (offline/dev testing) | disabled-by-default | P-1.1 | 2027-02-01 |
| `MLCOUNCIL_LLM_GGUF_PATH` | `""` | `models/sentiment_llm.py` | Local llama-cpp GGUF model path | param | P-1.1 | 2027-02-01 |
| `MLCOUNCIL_CS_FEATURE_ZSCORE` | `false` | `models/technical.py` | Cross-sectional z-score transform of features | disabled-by-default | P-1.1 | 2027-02-01 |
| `MLCOUNCIL_REGIME_FEATURES` | `false` | `models/regime_features.py` | Regime-conditioned feature augmentation for training | disabled-by-default | P-1.1 | 2027-02-01 |

## P-1.2 — execution/crypto (expiry **2027-06-01**)

| Flag | Default | Module | Purpose | Status | Target phase | Expiry |
|---|---|---|---|---|---|---|
| `MLCOUNCIL_DYNAMIC_SLIPPAGE` | `false` | `council/transaction_costs.py` | Notional-dependent slippage model | canary-controlled (G1: `dynamic_slippage`) | P-1.2 | 2027-06-01 |
| `MLCOUNCIL_SQRT_MARKET_IMPACT` | `false` | `council/transaction_costs.py` | Square-root market impact model | disabled-by-default | P-1.2 | 2027-06-01 |
| `MLCOUNCIL_SQRT_IMPACT_ETA` | `1.0` | `council/transaction_costs.py` | Impact coefficient η for sqrt model | param | P-1.2 | 2027-06-01 |
| `MLCOUNCIL_CRYPTO_ENABLED` | `false` (runtime.env: `true`) | `council/portfolio.py`, `intraday/supervisor.py` | Enable crypto universe in portfolio/intraday | disabled-by-default | P-1.2 | 2027-06-01 |

## P-2 — autonomy/risk (expiry **2027-12-01**)

| Flag | Default | Module | Purpose | Status | Target phase | Expiry |
|---|---|---|---|---|---|---|
| `MLCOUNCIL_CAUSAL_DRIFT_ENABLED` | `false` | `council/causal_drift.py` | PCMCI-style causal drift check in monitor | disabled-by-default | P-2 | 2027-12-01 |
| `MLCOUNCIL_GENERATIVE_STRESS` | `false` | `council/generative_stress.py` | Generative stress scenarios for VaR/CVaR | disabled-by-default | P-2 | 2027-12-01 |
| `MLCOUNCIL_RL_EXECUTION_ENABLED` | `false` | `execution/rl_agent.py` | PPO RL execution agent | disabled-by-default | P-2 | 2027-12-01 |
| `MLCOUNCIL_SMART_ROUTING_ENABLED` | `false` | `execution/router.py` | Smart order routing across venues | disabled-by-default | P-2 | 2027-12-01 |
| `MLCOUNCIL_MICROSTRUCTURE_PROMOTED` | `false` | `models/microstructure.py` | Wire OFI microstructure alpha into council | disabled-by-default | P-2 | 2027-12-01 |
| `MLCOUNCIL_MICROSTRUCTURE_SHADOW` | `true` | `models/microstructure.py` | Microstructure shadow logging (on by default) | shadow (default on) | P-2 | 2027-12-01 |
| `MLCOUNCIL_ORDERBOOK_FEED` | `synthetic` | `models/microstructure.py` | L2 order book feed source | param | P-2 | 2027-12-01 |

## P-3 — 2030 (expiry **2028-06-01**)

| Flag | Default | Module | Purpose | Status | Target phase | Expiry |
|---|---|---|---|---|---|---|
| `MLCOUNCIL_OPTIONS_SENTIMENT` | `false` | `models/options_sentiment.py` | Options-implied sentiment scaffold (put/call, skew) | disabled-by-default | P-3 | 2028-06-01 |

## Active/ops — no expiry

| Flag | Default | Module | Purpose |
|---|---|---|---|
| `MLCOUNCIL_ENV_PROFILE` | `local` | `runtime_env.py` | Runtime profile (`local`/`paper`/`prod`/`frontier`) |
| `MLCOUNCIL_RUNTIME_ENV_PATH` | `config/runtime.env` | `runtime_env.py` | Runtime env file path override |
| `MLCOUNCIL_DOTENV_PATH` | `./.env` | `runtime_env.py` | Project `.env` path override |
| `MLCOUNCIL_USE_PRODUCTION_MANIFEST` | `false` (prod: `true`) | `council/production_config.py` | Use `config/production_manifest.yaml` champions |
| `MLCOUNCIL_OTEL_ENABLED` | `false` | `observability/tracing.py` | OpenTelemetry tracing toggle |
| `MLCOUNCIL_OTEL_NAMESPACE` | `mlcouncil` | `observability/tracing.py` | OTel service namespace |
| `MLCOUNCIL_OTEL_SERVICE_NAME` | — | `observability/tracing.py` | OTel service name override |
| `MLCOUNCIL_TDA_WARNING_ENABLED` | `true` | `council/tda_warning.py` | TDA early-warning signal asset |
| `MLCOUNCIL_ONLINE_IC_THRESHOLD` | `0.05` | `models/online.py` | Online refit IC gate (param of online learning) |
| `MLCOUNCIL_ONLINE_REFIT_DAYS` | `60` | `models/online.py` | Online refit lookback (param) |
| `MLCOUNCIL_ONLINE_EVAL_DAYS` | `10` | `models/online.py` | Online eval window (param) |
| `MLCOUNCIL_IC_SHARPE_HALFLIFE` | `60` | `council/aggregator.py` | EWM halflife for IC-Sharpe performance weighting |
| `MLCOUNCIL_COVARIANCE_WINDOW` | `90` | `council/covariance_dynamic.py` | Covariance estimation window |
| `MLCOUNCIL_COMMISSION_BPS` | `0.5` | `council/transaction_costs.py` | Commission bps per side |
| `MLCOUNCIL_SLIPPAGE_BPS` | `5.0` | `council/transaction_costs.py` | Static slippage bps per side |
| `MLCOUNCIL_COST_CALIBRATION_PATH` | — | `council/transaction_costs.py` | Cost calibration JSON path |
| `MLCOUNCIL_COST_CALIBRATION_CONFIDENCE_FLOOR` | — | `council/transaction_costs.py` | Calibration confidence floor (%) |
| `MLCOUNCIL_MAX_DAILY_ORDERS` | `20` | `runtime_env.py`, `api/services/trading_service.py` | Max orders per day |
| `MLCOUNCIL_MAX_TURNOVER` | `0.30` | `runtime_env.py`, `council/portfolio.py` | Max one-way turnover |
| `MLCOUNCIL_MAX_POSITION_SIZE` | `0.10` | `runtime_env.py`, `council/portfolio.py`, `council/risk_engine.py` | Max single position size |
| `MLCOUNCIL_MAX_SECTOR_EXPOSURE` | `0.25` | `runtime_env.py`, `council/portfolio.py` | Max sector exposure |
| `MLCOUNCIL_MAX_CRYPTO_POSITION_SIZE` | `0.20` | `council/portfolio.py`, `council/risk_engine.py` | Max crypto position size |
| `MLCOUNCIL_MAX_CRYPTO_TURNOVER` | `0.40` | `council/portfolio.py`, `api/services/trading_service.py` | Max crypto turnover |
| `MLCOUNCIL_AUTOMATION_PAUSED` | `false` | `runtime_env.py`, `api/services/trading_service.py` | Pause automation (paper/prod guard) |
| `MLCOUNCIL_MIN_SIGNAL_STRENGTH` | `0.20` | `council/portfolio.py` | Min |z| signal strength |
| `MLCOUNCIL_MAX_DRAWDOWN_PCT` | `0.07` | `council/portfolio.py` | Drawdown circuit breaker |
| `MLCOUNCIL_MAX_VOL_ANN` | `0.30` | `council/portfolio.py` | Annualized vol cap |
| `MLCOUNCIL_MAX_VOL_DAILY` | `0.025` | `council/portfolio.py` | Daily vol cap |
| `MLCOUNCIL_TC_LAMBDA` | `2.0` | `council/portfolio.py` | TC penalty weight in objective |
| `MLCOUNCIL_RISK_LAMBDA` | `1/max_vol_daily²` | `council/portfolio.py` | Risk penalty weight |
| `MLCOUNCIL_PORTFOLIO_SHRINK_COV` | `true` | `council/portfolio.py` | Ledoit-Wolf shrinkage of covariance |
| `MLCOUNCIL_REBALANCE_EVERY` | `5` | `config/runtime.env` | Rebalance cadence (days) |
| `MLCOUNCIL_REGIME_MODE` | `label` | `council/aggregator.py` | Regime conditioning mode (`label`/`embedding`) |
| `MLCOUNCIL_TARGET_MODE` | `forward_return` | `data/features/target.py` | Training target mode (training/backtest flows only) |
| `MLCOUNCIL_TB_K` | `2.0` | `data/features/target.py` | Triple-barrier multiplier (param) |
| `MLCOUNCIL_TB_VOL_WINDOW` | `21` | `data/features/target.py` | Triple-barrier vol window (param) |
| `MLCOUNCIL_CORRELATION_THRESHOLD` | `0.7` | `council/risk_engine.py` | Correlation pair threshold |
| `MLCOUNCIL_MAX_CORRELATED_PAIRS` | `0.4` | `council/risk_engine.py` | Max correlated pairs fraction |
| `MLCOUNCIL_ALLOW_UNHASHED_PICKLE` | `false` | `council/pickle_security.py` | Allow unhashed pickle artifacts (security escape) |
| `MLCOUNCIL_ALLOWED_ORIGINS` | `http://localhost:8501` | `api/main.py` | Admin CORS origins |
| `MLCOUNCIL_API_KEY` | — | `api/auth.py` | Admin API key (required for paper/prod) |
| `MLCOUNCIL_REQUIRE_API_KEY` | auto (paper) | `api/auth.py` | Force API key requirement |
| `MLCOUNCIL_AUTO_EXECUTE` | `false` | `api/services/pipeline_automation.py` | Auto-execute pipeline runs |
| `MLCOUNCIL_AUTO_EXECUTE_POLL_SECONDS` | `5` | `api/services/pipeline_automation.py` | Pipeline automation poll interval |
| `MLCOUNCIL_AUTO_PROMOTE_MODELS` | `false` | `data/pipeline.py` | Auto-promote champion after gate passes |
| `MLCOUNCIL_INTRADAY_UNIVERSE` | — | `api/services/intraday_runtime_service.py` | Intraday universe (tickers/crypto) |
| `MLCOUNCIL_INTRADAY_INTERVAL_MINUTES` | `15` | `api/services/intraday_runtime_service.py` | Intraday scheduling interval |
| `MLCOUNCIL_INTRADAY_AGENT_PROVIDER` | `rule-based` | `api/services/intraday_runtime_service.py` | Intraday agent provider |
| `MLCOUNCIL_INTRADAY_LOG_TO_MLFLOW` | `false` | `api/services/intraday_runtime_service.py` | Log intraday runs to MLflow |
| `MLCOUNCIL_INTRADAY_MIN_VALID_CLOSE_RATIO` | `0.70` | `intraday/supervisor.py` | Intraday data quality gate |
| `MLCOUNCIL_INTRADAY_MIN_INFORMATIVE_RATIO` | `0.50` | `intraday/supervisor.py` | Intraday data quality gate |
| `MLCOUNCIL_OPENAI_INTRADAY_MODEL` | `gpt-4o-mini` | `intraday/agent.py` | LLM model for intraday agent |
| `MLCOUNCIL_PRIVATE_ASSISTANT_PATCH` | — | `scripts/patch_mlflow_assistant.py` | Marker for private MLflow assistant patch script |
| `MLCOUNCIL_MAX_VAR_PCT` | — (RiskLimits attr `0.015`) | `council/risk_engine.py` | VaR limit — documented; no `os.getenv` read found |
| `MLCOUNCIL_MAX_CVAR_PCT` | — (RiskLimits attr `0.025`) | `council/risk_engine.py` | CVaR limit — documented; no `os.getenv` read found |

## Doc-only / removal candidates

| Flag | Default | Module | Purpose | Status |
|---|---|---|---|---|
| `MLCOUNCIL_VECTOR_STORE_MOCK` | — | `docs/adr/2026-05-21-finma-rag-sentiment.md` | Force in-memory vector store mock (RAG sentiment ADR) | doc-only — no code read found; removal candidate per Regole |

## Regole

1. **Telemetry rule** — un flag senza telemetria nel daily path = candidato alla
   rimozione. Flags listed above as `doc-only` or with "no `os.getenv` read
   found" (`MLCOUNCIL_MAX_VAR_PCT`, `MLCOUNCIL_MAX_CVAR_PCT`,
   `MLCOUNCIL_VECTOR_STORE_MOCK`) must be either wired with telemetry or
   removed at the next F-0.4 review.
2. **Expiry = decision date** — ogni flag `disabled-by-default` ha una data di
   *decisione* promote-o-retire (canary `council/canary.py` + `config/canary.yaml`
   quando il G1 gate approva l'attivazione). Superata la data senza promozione,
   il flag va rimosso o rifinanziato in una fase successiva.
3. **Registry update cadence** — il registro va aggiornato a ogni fase (P-1.1 →
   P-1.2 → P-2 → P-3): nuovi flag aggiunti, flag promossi spostati in
   active/ops, flag ritirati rimossi dal codice e dalla tabella.
4. **Docstring annotation** — ogni flag disabled-by-default ha una riga finale
   nel docstring del modulo proprietario nel formato:
   `Canary status: shadow — target: <fase> — expiry: <data> (promote via canary o retire)`.
5. **Canary precedence** — `council/canary.py` usa `os.environ.setdefault`:
   un env esplicito dell'operatore (o il production manifest) vince sul valore
   canary. Il revert è sticky (kill switch, P4).
