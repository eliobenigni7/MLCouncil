# Model Promotion Criteria

## Obiettivo

Un candidato puo essere promosso solo se il suo training, backtest e validazione sono auditabili e migliori del baseline minimo operativo.

## Requisiti minimi

- run MLflow presente e leggibile
- lineage completo:
  - `pipeline_run_id`
  - `data_version`
  - `feature_version`
  - `model_version`
- metriche minime registrate:
  - `sharpe`
  - `max_drawdown`
  - `turnover`
  - `oos_sharpe`
  - `oos_max_drawdown`
  - `oos_turnover`
  - `walk_forward_window_count`
  - `pbo`

## Gate operativi

Il candidato non si promuove se:

- `oos_sharpe <= 0`
- `pbo > 0.50`
- `walk_forward_window_count < 1`
- mancano tag MLflow standardizzati
- manca il riferimento a dati o feature versionate

## Check qualitativi

Prima della promozione verificare:

1. le metriche `gross` e `net` sono entrambe presenti
2. `estimated_costs_usd` e coerente con il turnover
3. non ci sono fallback silenziosi nel training o nel backtest
4. la strategia non richiede override manuali per passare il gate

## Checklist rapida

- training completato con run MLflow
- backtest completato con metriche `gross` e `net`
- diagnostica walk-forward registrata
- promotion gate verde
- documentazione del candidato aggiornata
- nessun blocker aperto nei controlli operativi Fase 3-4
