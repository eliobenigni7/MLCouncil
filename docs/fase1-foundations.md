# Fase 1 Foundations

## Cosa cambia

- I principali asset Dagster hanno un data contract esplicito e un asset check bloccante.
- La pipeline propaga il lineage minimo richiesto: `pipeline_run_id`, `data_version`, `feature_version`, `model_version`.
- Gli ordini giornalieri persistono il lineage sia nel parquet sia nella risposta dell'esecuzione trading.
- MLflow usa un contratto condiviso per naming, tag obbligatori e promotion gate minimo.
- La CI esegue `pytest`, smoke import della pipeline Dagster e build Docker.

## Convenzioni di lineage

- `pipeline_run_id`: `context.run_id` Dagster, con fallback `manual-<partition_date>`.
- `data_version`: hash stabile del payload sorgente rilevante per l'asset.
- `feature_version`: hash stabile del payload di feature o segnale.
- `model_version`: hash del checkpoint o versione esplicita del modello.

## MLflow

- Tag obbligatori per training e backtest:
  - `model_name`
  - `pipeline_run_id`
  - `data_version`
  - `feature_version`
  - `environment`
- Metriche minime per il promotion gate:
  - `sharpe`
  - `max_drawdown`
  - `turnover`

## Runtime env

- Profilo default: `local`
- Override esplicito: `MLCOUNCIL_RUNTIME_ENV_PATH=/path/to/runtime.env`
- Profili supportati:
  - `config/runtime.local.env`
  - `config/runtime.paper.env`
- File example:
  - `config/runtime.local.env.example`
  - `config/runtime.paper.env.example`
- `config/runtime.env` resta solo come fallback legacy condiviso.
