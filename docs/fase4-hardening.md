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

## Effetto operativo

- la separazione `local` vs `paper` e verificabile a runtime
- il profilo `paper` fallisce in health se mancano chiavi o se l'endpoint Alpaca non e paper
- lo stato dell'ultima operazione di trading compare in health
- i moduli critici prima non coperti hanno ora regression tests mirati

## Limiti noti ancora aperti

- `api/main.py` usa ancora `on_event`, deprecato in FastAPI
- `council/alerts.py` usa ancora `datetime.utcnow()`, deprecato in Python moderno
- il cleanup finale di `pytest` su Windows puo loggare un warning di permessi sulla cartella temp
