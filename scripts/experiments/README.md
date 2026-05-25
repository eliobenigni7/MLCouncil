# Script sperimentali MLCouncil

Questa directory contiene varianti, spike e runner diagnostici.

## Non canonici

Non usarli come singola fonte di verità per i backtest ufficiali.

File spostati qui:

- `playground_batch.py`
- `playground_batch_v2.py`
- `playground_batch_v3.py`
- `quick_test_playground.py`
- `regime_gate_backtest.py`
- `run_optimized_backtest.py`
- `tft_one_year_backtest.py`
- `run_sharpe_ablation.py`
- `train_tft.py`
- `train_regime_dss.py`
- `train_moe_gating.py`
- `train_stacking_cqr.py`
- `train_alpha_portfolio_end2end.py`

I path root `scripts/run_*.py` e `scripts/train_*.py` per questi tool restano disponibili come wrapper di compatibilità.

## Regola operativa

Se uno di questi script diventa affidabile e ripetibile, va:

1. promosso fuori da `scripts/experiments/`
2. coperto da test/regression check
3. documentato nel canale canonico
