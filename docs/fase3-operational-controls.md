# Fase 3 Operational Controls

La Fase 3 rende il paper trading operabile con controlli espliciti lungo il percorso ordini.

## Cosa cambia

- `TradingService` costruisce un preflight unico prima dell'esecuzione.
- Il preflight verifica:
  - paper-mode enforcement
  - kill switch `MLCOUNCIL_AUTOMATION_PAUSED`
  - cap giornaliero ordini
  - turnover massimo
  - breach di rischio via `RiskEngine`
  - reconciliation tra target della pipeline e posizioni correnti
- Gli ordini salvati in `data/orders/{date}.parquet` vengono interpretati come notionals USD quando includono `target_weight`; prima dell'invio vengono convertiti in share quantity usando il prezzo corrente.

## Endpoint nuovi o estesi

- `GET /api/trading/status`
  - include `runtime_profile`, `paused`, `kill_switch_active`, `paper_guard_ok`
- `GET /api/trading/preflight/{date}`
  - restituisce snapshot operativo completo prima dell'invio ordini
- `GET /api/trading/reconcile/{date}`
  - restituisce la riconciliazione target vs posizioni correnti
- `POST /api/trading/execute`
  - risponde con `409` quando il preflight blocca il run

## Artifact operativi

- `data/risk/risk_report_{date}.json`
  - report di rischio del portafoglio proiettato
- `data/operations/{date}.json`
  - stato operativo del run di trading
- `data/paper_trades/{date}.json`
  - log di submission e liquidazioni

## Variabili operative

- `MLCOUNCIL_AUTOMATION_PAUSED`
- `MLCOUNCIL_MAX_DAILY_ORDERS`
- `MLCOUNCIL_MAX_TURNOVER`
- `MLCOUNCIL_MAX_POSITION_SIZE`

## Note operative

- Il kill switch ferma solo l'esecuzione paper; la pipeline analitica puo continuare a girare.
- I breach `HIGH` di `RiskEngine` bloccano l'invio.
- I rejection per singolo ordine non bloccano il run se il preflight e verde, ma vengono loggati come stato `degraded`.
