# MLCouncil — triage operativo del repo

Questo file separa **canale canonico**, **sperimentazione** e **backup** per evitare che i flussi si mischino di nuovo.

## Canale canonico

Usare questi entrypoint come riferimento principale:

- `scripts/run_strategy_backtest.py` — backtest end-to-end coerente con il flusso reale del progetto
- `scripts/one_year_backtest.py` — runner diagnostico / iterativo a finestra corta
- `scripts/run_pipeline.py` — pipeline operativa / demo standalone
- `backtest/runner.py`, `backtest/simulator.py`, `backtest/validation.py` — core logico del backtest

## Sperimentale / prototipo

Tutti gli script spostati in `scripts/experiments/` sono prototipi o varianti da non considerare canoniche.

Regola pratica:

- se serve per verificare un'ipotesi → `scripts/experiments/`
- se serve per metriche affidabili o baseline → canale canonico
- se un prototipo diventa stabile → promuoverlo fuori da `scripts/experiments/` e aggiungere regressioni

## Documenti sperimentali

I documenti di proposta o wave/roadmap non canonici vanno in `docs/experiments/`.

## Backup e artefatti temporanei

I backup manuali o file `.bak` non devono stare nel root del repo. Vanno in `config/backups/` oppure in una directory di artefatti equivalente.

## Criterio di promozione

Un file esce dall'area sperimentale solo se:

1. ha un ruolo chiaramente definito
2. ha test di regressione o smoke test
3. non sovrascrive config live o risultati canonici
4. non duplica un entrypoint già esistente

## Stato attuale del triage

- esperimenti backtest isolati in `scripts/experiments/`
- proposta wave3 isolata in `docs/experiments/`
- backup YAML isolato in `config/backups/`
- core backtest lasciato intatto
