# Fase 4 Hardening

La Fase 4 chiude i gap piu rischiosi rimasti dopo le Fasi 1-3.

## Deliverable

- validazione esplicita dei profili runtime in `runtime_env.py`
- health API estesa con:
  - stato `runtime_env`
  - stato `trading_operations`
- test aggiunti per:
  - `runtime_env.py`
  - `execution/alpaca_adapter.py`
  - `data/store/arctic_store.py`
- runbook operativo giornaliero
- criteri di promozione modello documentati
- quality gates CI/CD incrementali:
  - coverage gate `pytest --cov` con soglia iniziale 68%
  - lint gate `ruff` su moduli critici (`api/main.py`, `api/auth.py`, `api/services/trading_service.py`, `runtime_env.py`, `council/portfolio.py`)
  - type-check gate `mypy` (scope incrementale sugli stessi moduli critici)
  - security gates: secret scan statico, `pip-audit` su `requirements.txt`, `bandit -lll` su `api/`, `council/`, `execution/`, `runtime_env.py`

## Effetto operativo

- la separazione `local` vs `paper` e verificabile a runtime
- il profilo `paper` fallisce in health se mancano chiavi o se l'endpoint Alpaca non e paper
- lo stato dell'ultima operazione di trading compare in health
- i moduli critici prima non coperti hanno ora regression tests mirati
- ogni PR deve passare job separati (`tests-and-coverage`, `lint`, `typecheck`, `security`) prima del build Docker

## Limiti noti ancora aperti

- `api/main.py` usa ancora `on_event`, deprecato in FastAPI
- `council/alerts.py` usa ancora `datetime.utcnow()`, deprecato in Python moderno
- il cleanup finale di `pytest` su Windows puo loggare un warning di permessi sulla cartella temp
- il type-check e ancora a copertura incrementale (non full-repo) per evitare blocchi su moduli storici con dipendenze non tipizzate
