# MLCouncil — triage operativo del repo

Questo file separa **canale canonico**, **sperimentazione** e **backup** per evitare che i flussi si mischino di nuovo.

## Canale canonico

Usare questi entrypoint come riferimento principale:

- `scripts/run_strategy_backtest.py` — backtest end-to-end coerente con il flusso reale del progetto
- `scripts/one_year_backtest.py` — runner diagnostico / iterativo a finestra corta
- `scripts/run_pipeline.py` — pipeline operativa / demo standalone
- `backtest/runner.py`, `backtest/simulator.py`, `backtest/validation.py` — core logico del backtest

## Orchestrazione / servizio

Runner canonici che non vanno confusi con esperimenti o ablation:

- `scripts/run_walkforward_promotion.py` — orchestratore di retrain + gate per promotion CI
- `scripts/run_intraday_supervisor.py` — entrypoint di servizio per il supervisor intraday

## Setup / promotion ops

Script operativi canonici o di bootstrap che restano fuori da `scripts/experiments/`:

- `scripts/setup_prod.py` — bootstrap della production manifest e dei placeholder di gate
- `scripts/populate_walkforward_caches.py` — seed dei cache parquet per CI / gate locale
- `scripts/promote_model.py` — promozione operatore modello gated verso production manifest
- `scripts/promote_council_module.py` — promozione operatore per moduli council/portfolio
- `scripts/establish_wave2_staging_promotion.py` — helper di staging locale per TFT / walk-forward
- `scripts/bootstrap_frontier.py` — bootstrap one-shot del profilo frontier / R&D

## Training

Entry point di training da tenere distinti dal resto:

- canonico: `scripts/train_lgbm_standalone.py`, `scripts/train_meta_label.py`
- shadow / challenger: `scripts/experiments/train_tft.py`, `scripts/experiments/train_regime_dss.py`, `scripts/experiments/train_moe_gating.py`, `scripts/experiments/train_stacking_cqr.py`, `scripts/experiments/train_alpha_portfolio_end2end.py`

I path root `scripts/train_*.py` per gli script shadow restano come wrapper di compatibilità, ma il codice vero vive in `scripts/experiments/`.

## Sperimentale / prototipo

Tutti gli script spostati in `scripts/experiments/` sono prototipi o varianti da non considerare canoniche.

Regola pratica:

- se serve per verificare un'ipotesi → `scripts/experiments/`
- se serve per metriche affidabili o baseline → canale canonico
- se un prototipo diventa stabile → promuoverlo fuori da `scripts/experiments/` e aggiungere regressioni

## Documenti sperimentali

I documenti di proposta o wave/roadmap non canonici vanno in `docs/internal/experiments/`.

## Backup e artefatti temporanei

I backup manuali o file `.bak` non devono stare nel root del repo. Vanno in `config/backups/` oppure in una directory di artefatti equivalente.

## Criterio di promozione

Un file esce dall'area sperimentale solo se:

1. ha un ruolo chiaramente definito
2. ha test di regressione o smoke test
3. non sovrascrive config live o risultati canonici
4. non duplica un entrypoint già esistente

## Stato attuale del triage

- esperimenti backtest e ablation isolati in `scripts/experiments/`
- proposta wave3 isolata in `docs/internal/experiments/`
- backup YAML isolato in `config/backups/`
- core backtest lasciato intatto
