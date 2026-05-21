# Clean Baseline — 2026-05-21

Post-P0 reference measurement for MLCouncil walk-forward validation, benchmark comparison, regime breakdown, gross/net costs, and stage runtime. Future TO BE challengers should beat these numbers under the same protocol (or document protocol changes).

**Status:** Completed on local workspace `E:\Github\MLCouncil`. OHLCV raw data was available; no Dagster/materialization run was required for this baseline.

---

## Provenance

| Field | Value |
|---|---|
| **Date** | 2026-05-21 |
| **Git SHA** | `12276e8cedae4c430a44ab24274e4192cb47d7d9` |
| **Branch** | `master` (tracking `origin/master`) |
| **Config hash** | `292e779b9d086def` (SHA-256 of `config/universe.yaml`, `config/models.yaml`, `config/runtime.env`) |
| **Python** | 3.14.0 (Windows 11 10.0.26200) |
| **Key packages** | lightgbm 4.6.0, polars 1.39.3, pytest 9.0.2 |
| **Not installed** | cvxpy (portfolio optimizer fell back to sector-aware weights) |

---

## Commands run

### 1. End-to-end strategy backtest (primary baseline)

```powershell
Set-Location E:\Github\MLCouncil
$env:PYTHONIOENCODING = 'utf-8'
New-Item -ItemType Directory -Force -Path data\results | Out-Null
python scripts/run_strategy_backtest.py
```

**Wall time:** ~499 s (~8.3 min) on this machine.

**Artifacts:** `data/results/walk_forward_summary.json`, `walk_forward_benchmark.parquet`, `walk_forward_regime.parquet`, `walk_forward_windows.parquet`, `equity_curve.parquet`, `strategy_weights.parquet`, and related files.

### 2. Verification (required)

```powershell
python -m pytest tests/test_backtest_validation.py tests/test_retraining.py -v
```

**Result:** 16 passed in 2.51s.

### 3. Config hash helper (reproducibility)

```powershell
python -c "import hashlib; from pathlib import Path
ROOT=Path('.')
h=hashlib.sha256()
for name in ['universe.yaml','models.yaml','runtime.env']:
    p=ROOT/'config'/name
    if p.exists():
        h.update(name.encode()); h.update(p.read_bytes())
print(h.hexdigest()[:16])"
```

---

## Data coverage

| Item | Value |
|---|---|
| **OHLCV source** | `data/raw/ohlcv/{TICKER}/*.parquet` |
| **Tickers with parquet** | 22 directories |
| **Global date range (raw)** | 2021-01-04 → 2025-12-31 |
| **Strategy weight universe** | 10 tickers: AAPL, AMZN, GOOGL, JPM, JNJ, META, MSFT, NVDA, V, XOM |
| **Strategy weight dates** | 2021-07-22 → 2026-01-27 (1,134 rebalance days) |
| **Feature rows / calendar days** | 13,142 rows / 1,315 dates |
| **Walk-forward splits built** | 18 (3 windows skipped empty train/test) |
| **Walk-forward windows in metrics** | 15 |
| **Train / test window** | 131 / 63 trading days |
| **Purge / embargo** | 1 / 1 day |

Macro inputs used when present: `data/raw/macro/{vix,treasuries,sp500}.parquet`.

---

## Runtime by stage

Timestamps taken from run log (`2026-05-21`).

| Stage | Description | Approx. duration |
|---|---|---|
| **[1/7] Load data** | OHLCV load, Alpha158, targets, macro, split planning | ~4 s |
| **[2/7] Walk-forward train** | 18 windows × (LightGBM fit, HMM, conformal, daily council + portfolio) | ~485 s |
| **[3/7] Simulate portfolio** | `simulate_weight_backtest` with `TransactionCostModel.from_env()` | &lt;1 s |
| **[4/7] Persist artifacts** | Parquet/JSON/pickle writes | ~5 s |
| **Total** | | **~499 s** |

Dominant cost: per-window LightGBM retrain and daily inference loop, not simulation I/O.

---

## Metrics summary

### A. Full-period portfolio simulation (`simulate_weight_backtest`)

Uses produced `strategy_weights` and forward returns; costs from `MLCOUNCIL_COMMISSION_BPS=1.0`, `MLCOUNCIL_SLIPPAGE_BPS=3.0` in `config/runtime.env`.

| Metric | Net | Gross |
|---|---:|---:|
| Sharpe | 0.99 | 1.02 |
| CAGR | 19.9% | 20.8% |
| Max drawdown | -28.0% | -27.7% |
| Calmar | 0.71 | 0.75 |
| Turnover (avg) | 7.1% | — |
| Estimated costs (USD) | 4,245 | — |
| Final equity ($100k start) | 226,342 | 233,711 |
| Horizon | ~4.5 years | |

**Gross vs net gap:** ~3.2% CAGR and ~0.03 Sharpe from heuristic transaction costs over the full backtest path.

### B. Walk-forward OOS diagnostics (`run_walk_forward_analysis` on council signals)

Purged-embargoed windows; OOS metrics averaged across test windows. **Not identical** to Section A (signal-return proxy vs weight-based simulation).

| Metric | Value |
|---|---:|
| `walk_forward_window_count` | 15 |
| `oos_sharpe` | 0.042 |
| `oos_max_drawdown` | -6.0% (mean window) |
| `oos_turnover` | 46.7% (signal-based estimate) |
| `pbo` | 0.267 |
| `equal_weight_sharpe_delta` | -1.34 |
| `equal_weight_cagr_delta` | -0.29 |
| `regime_count` | 3 |

### C. Benchmark comparison (OOS vs suite)

From `data/results/walk_forward_benchmark.parquet`. All four core benchmarks from `build_benchmark_suite` are present.

| Benchmark | Bench. Sharpe | Strategy Sharpe | Sharpe Δ | Bench. CAGR | Strategy CAGR | CAGR Δ |
|---|---:|---:|---:|---:|---:|---:|
| equal_weight | 1.14 | -0.20 | -1.34 | 25.3% | -3.2% | -28.5% |
| momentum_long_only | 1.32 | -0.20 | -1.52 | 32.5% | -3.2% | -35.7% |
| inverse_volatility | 1.07 | -0.20 | -1.27 | 20.3% | -3.2% | -23.5% |
| vol_target_equal_weight | 1.42 | -0.20 | -1.62 | 25.0% | -3.2% | -28.2% |

On this OOS signal path, the strategy underperformed all passive benchmarks. Section A simulation Sharpe is higher because it uses portfolio weights, rebalance cadence, conformal sizing, and a different return construction — document both when comparing challengers.

### D. Regime breakdown (OOS)

From `data/results/walk_forward_regime.parquet` (labels derived from equal-weight benchmark returns).

| Regime | n_obs | Sharpe | CAGR | Max DD |
|---|---:|---:|---:|---:|
| bull | 378 | -0.28 | -3.1% | -15.8% |
| transition | 350 | 0.58 | 7.1% | -17.2% |
| bear | 217 | -1.29 | -17.9% | -18.4% |

### E. Ablation

`walk_forward_ablation.parquet` is **empty** for this run: `scripts/run_strategy_backtest.py` does not pass `component_signals` into `run_walk_forward_analysis`. Ablation is available in unit tests and can be wired in a follow-up baseline refresh.

---

## Blockers and caveats

| ID | Severity | Note |
|---|---|---|
| B1 | High | **cvxpy not installed** — live-style script used sector-aware portfolio fallback, not CVXPY mean-variance optimizer documented in README. |
| B2 | High | **Two metric paths** — WF OOS signal Sharpe (~0.04) vs full simulation Sharpe (~0.99) measure different objects; do not merge without relabeling. |
| B3 | Medium | **Partial universe** — 10/22 available tickers in weights; config lists more names than local OHLCV. |
| B4 | Medium | **No sentiment in script path** — daily backtest uses LightGBM + HMM regime only (`hmm` signal zeroed in council). |
| B5 | Medium | **Unicode console** — first run failed on Windows cp1252 for `→` in log lines; rerun with `PYTHONIOENCODING=utf-8`. |
| B6 | Low | **Ablation not populated** — see Section E. |
| B7 | Low | **P0 doc drift** — architecture doc still lists P1 baseline as pending; this file satisfies that measurement for 2026-05-21. |

**Not blocked:** raw OHLCV, macro parquet, walk-forward validation module, benchmark suite, regime summary, gross/net simulation, pytest verification.

**Not run:** Dagster daily pipeline, NautilusTrader `backtest/runner.py`, ArcticDB materialization, paper trading execution, MLflow logging.

---

## Regeneration checklist

1. Ensure `data/raw/ohlcv` covers intended universe and dates.
2. Install optional `cvxpy` if optimizer parity with production is required.
3. Set `PYTHONIOENCODING=utf-8` on Windows.
4. Run `python scripts/run_strategy_backtest.py`.
5. Run verification pytest command above.
6. Update git SHA, config hash, and tables in this file.

---

## References

- Walk-forward implementation: `backtest/validation.py`, `scripts/run_strategy_backtest.py`
- Benchmark suite: `build_benchmark_suite` (equal_weight, momentum_long_only, inverse_volatility, vol_target_equal_weight)
- Architecture context: `docs/architecture-as-is-to-be-2026-05-21.md`, `AGENTS.md`
- Prompt: `docs/agentic-prompts-2026-05-21.md` (Prompt 08)
