# Fase 5 Governance + Operator UX

La Fase 5 chiude i gap di auditabilita e disciplina operativa dopo hardening CI.

## Deliverable

- policy standard per integrita artefatti:
  - sidecar `.manifest` con `sha256`, dimensione, profilo runtime, fingerprint dipendenze, git sha
  - applicata a checkpoint modello, ordini giornalieri, report operativi, output regime/attribution, report rischio
- contratti estesi (`data/contracts.py`) per payload critici:
  - `council_signal`
  - `portfolio_weights`
  - `operations_report`
  - `risk_report`
  - `backtest_metrics`
  - `model_evaluation_metrics`
- validazioni runtime aggiuntive:
  - metriche backtest validate prima del logging MLflow
  - metriche candidate validate nel retraining prima del promotion gate
  - report operativi/risk validate prima della persistenza
- governance processo:
  - `.github/pull_request_template.md`
  - `.github/CODEOWNERS`
  - issue templates (`bug_report`, `model_promotion`)
  - ADR template + indice (`docs/adr/`)
- operator-safe UX in Admin:
  - banner stato operativo (runtime invalid / trading degraded / kill switch)
  - conferma esplicita per `Run Pipeline` ed `Execute Orders`
  - KPI overview estesi con runtime profile e stato ultimo trade

## Effetto operativo

- ogni artifact critico e tracciabile con integrita verificabile
- i payload operativi hanno contratti espliciti e regressione automatica
- la review su superfici sensibili diventa strutturata (template + ownership)
- l'UI riduce azioni involontarie e rende piu visibili stati bloccanti

## Limiti noti ancora aperti

- `CODEOWNERS` usa ownership single-maintainer e puo essere raffinato quando il team si allarga
- i contratti estesi coprono i payload principali ma non tutti i file storici legacy
- i manifest non sono ancora verificati automaticamente in pipeline di promozione (solo prodotti a runtime)
