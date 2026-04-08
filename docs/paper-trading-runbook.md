# Paper Trading Runbook

## Obiettivo

Questo runbook descrive il flusso operativo minimo per gestire MLCouncil in modalita paper trading con controlli Fase 1-4 attivi.

## Pre-check giornaliero

1. Verificare il profilo runtime:
   - `MLCOUNCIL_ENV_PROFILE=paper`
   - `ALPACA_BASE_URL=https://paper-api.alpaca.markets`
   - chiavi `ALPACA_PAPER_KEY` e `ALPACA_PAPER_SECRET` presenti
2. Verificare il kill switch:
   - `MLCOUNCIL_AUTOMATION_PAUSED=false` per eseguire
   - `true` per bloccare l'invio ordini mantenendo attiva la pipeline analitica
3. Controllare health:
   - `GET /api/health`
   - `GET /api/trading/status`
4. Controllare l'ultima operazione:
   - `GET /api/trading/preflight/{date}`
   - `GET /api/trading/reconcile/{date}`

## Sequenza operativa

1. Eseguire o verificare la pipeline Dagster del giorno.
2. Verificare che `data/orders/{date}.parquet` esista e contenga lineage.
3. Chiamare `GET /api/trading/preflight/{date}`.
4. Se `pretrade.blocked=true`, non eseguire `POST /api/trading/execute`.
5. Se il preflight e verde, eseguire `POST /api/trading/execute`.
6. Controllare gli artifact:
   - `data/operations/{date}.json`
   - `data/paper_trades/{date}.json`
   - `data/risk/risk_report_{date}.json`

## Hard stop

Il run va fermato quando si verifica uno di questi casi:

- `paper_guard_ok=false`
- `paused=true`
- `pretrade.blocked=true`
- breach `HIGH` nel report di rischio
- `Projected turnover` oltre limite
- count ordini oltre `MLCOUNCIL_MAX_DAILY_ORDERS`

## Triage rapido

### Health degradato per `runtime_env`

- verificare il file `config/runtime.paper.env`
- verificare le chiavi mancanti in `runtime_env.missing`
- correggere `ALPACA_BASE_URL` se punta all'endpoint live

### Health degradato per `trading_operations`

- leggere `data/operations/{date}.json`
- se `trade_status=blocked`, verificare `pretrade.reason`
- se `trade_status=degraded`, verificare `orders_rejected` e trade log

### Ordini rifiutati

- controllare `target_weight`
- controllare stima prezzo e conversione notional -> shares
- verificare i cap `MLCOUNCIL_MAX_POSITION_SIZE` e `MLCOUNCIL_MAX_TURNOVER`

## Uso del kill switch

Impostare:

```env
MLCOUNCIL_AUTOMATION_PAUSED=true
```

Effetto:

- la pipeline continua a produrre dati e ordini
- il preflight trading blocca l'esecuzione paper
- `POST /api/trading/execute` restituisce `409`

## Post-run

1. Confermare che `trade_status` sia `success` o `degraded`.
2. Se `degraded`, aprire follow-up sui simboli rifiutati.
3. Se il run e stato manualmente sospeso, riportare `MLCOUNCIL_AUTOMATION_PAUSED=false` solo dopo aver chiuso la causa.
