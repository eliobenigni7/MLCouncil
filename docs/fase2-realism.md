# Fase 2 Realism

## Cosa cambia

- I costi di transazione usano un contratto condiviso in `council/transaction_costs.py`.
- `PortfolioConstructor` e `BacktestReport` stimano turnover e costi con la stessa convenzione.
- Il backtest conserva l'equity lorda ma usa l'equity netta come default operativo.
- Le metriche del runner e di MLflow espongono sia il lato `gross` sia il lato `net`.
- Il retraining aggiunge diagnostica out-of-sample deterministica: `oos_sharpe`, `oos_max_drawdown`, `oos_turnover`, `walk_forward_window_count`, `pbo`.
- Il promotion gate rifiuta candidati senza diagnostica walk-forward sufficiente, con `oos_sharpe <= 0` o con `pbo > 0.50`.

## Cost model

- `commission_bps` e `slippage_bps` confluiscono in `TransactionCostModel.total_cost_bps`.
- `estimate_cost_from_weights` usa il turnover one-way `sum(abs(dw)) / 2`.
- `estimate_cost_from_notional` usa direttamente il nozionale scambiato del fill report.

## Backtest e reportistica

- `BacktestReport.gross_equity_curve` conserva l'equity fornita o ricostruita dai fill.
- `BacktestReport.equity_curve` rappresenta il netto dopo costi stimati cumulati.
- `BacktestReport.total_estimated_cost_usd` rende esplicito il delta fra lordo e netto.
- `_compute_stats` nel runner deriva le metriche da `BacktestReport`, evitando doppi calcoli incoerenti.

## Walk-forward e PBO-style proxy

- `backtest/validation.py` costruisce split walk-forward deterministici su indici temporali.
- Sono disponibili split con `purge_period` ed `embargo_period` (`build_purged_walk_forward_splits`) per ridurre leakage vicino ai boundary temporali.
- `summarize_walk_forward_metrics` produce le metriche OOS aggregate richieste dal gate.
- `pbo` qui e' un proxy deterministico per CI e gating:
  - misura quanto spesso una forza in-sample sopra mediana corrisponde a una performance out-of-sample sotto mediana
  - non sostituisce una CSCV/PBO accademica completa
- `backtest.runner.run_walk_forward_backtest(...)` espone un runner unico per validazione retrospettiva con:
  - `window_metrics` per finestra
  - `benchmark_comparison` automatico (delta vs equal-weight o benchmark espliciti)
  - `regime_performance` per decomposizione bull/bear/transition
  - `environment_metadata` (python/platform/numpy/pandas/runtime profile) per riproducibilita'.

## Estensioni Fase 3 (validation depth)

- `run_walk_forward_analysis(...)` produce in un unico bundle:
  - metrica OOS aggregata
  - confronto benchmark
  - breakdown per regime
  - analisi `ablation_analysis` con delta marginale di Sharpe per componente
- Il retraining (`data/retraining.py`) propaga ora anche:
  - `equal_weight_sharpe_delta`
  - `equal_weight_cagr_delta`
  - `regime_count`
  insieme alle metriche walk-forward standard (`oos_sharpe`, `oos_max_drawdown`, `oos_turnover`, `walk_forward_window_count`, `pbo`).

## Benchmark suite

- `build_benchmark_suite(...)` costruisce benchmark deterministici da `forward_returns`:
  - `equal_weight`
  - `momentum_long_only`
  - `inverse_volatility`
  - `vol_target_equal_weight`
- `run_walk_forward_backtest(...)` puo' ricevere benchmark espliciti oppure usare questa suite in fallback.

## MLflow

- `log_backtest_result` salva:
  - `sharpe`, `max_drawdown`, `turnover`
  - `gross_sharpe`, `gross_max_drawdown`, `gross_cagr`, `gross_calmar`
  - `estimated_costs_usd`, `gross_final_equity`, `net_final_equity`
- Il tracking URI viene impostato esplicitamente anche nei log di backtest.
