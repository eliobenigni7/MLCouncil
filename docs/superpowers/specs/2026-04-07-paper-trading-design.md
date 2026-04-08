# Paper Trading via Admin UI — Design Spec

**Date:** 2026-04-07
**Status:** Draft

---

## 1. Overview

Add paper trading execution to the MLCouncil Admin UI. When the Dagster pipeline generates daily orders, they can be automatically executed against Alpaca Paper Trading via the UI, with safety guards enforced.

**User Story:** Pipeline generates orders → User reviews in UI → System auto-executes to Alpaca Paper → Trade log visible in UI.

---

## 2. Architecture

```
data/orders/{date}.parquet  (pipeline output)
            ↓
api/services/trading_service.py  (reads + validates orders)
            ↓
execution/alpaca_adapter.py  (submits to Alpaca Paper API)
            ↓
data/paper_trades/{date}.json  (local trade log)
            ↓
UI: /trading page  (status, positions, history)
```

---

## 3. Components

### 3.1 API Router: `api/routers/trading.py`

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/trading/status` | Alpaca connection, account info, positions |
| GET | `/trading/orders/pending` | Latest orders from parquet (unexecuted) |
| GET | `/trading/orders/latest` | Most recent order date available |
| POST | `/trading/execute` | Execute pending orders to Alpaca |
| POST | `/trading/liquidate` | Liquidate all positions |
| GET | `/trading/history` | Past executed trades |

### 3.2 Service: `api/services/trading_service.py`

**Functions:**
- `get_trading_status()` — Returns Alpaca connection state, account info, positions
- `get_pending_orders(date: str)` — Reads `data/orders/{date}.parquet`, returns list of orders with direction/quantity
- `execute_orders(date: str)` — Main execution logic
  1. Load pending orders
  2. Fetch current positions from Alpaca
  3. Compute diff: liquidate positions not in target
  4. Apply safety checks (see Section 4)
  5. Submit market orders via `AlpacaLiveNode`
  6. Log trades to `data/paper_trades/{date}.json`
  7. Return execution summary
- `liquidate_all()` — Liquidate all current positions
- `get_trade_history(days: int)` — Load trades from log files

### 3.3 UI: Admin HTML/JS

**New nav section:** "Trading" between "Portfolio" and "Monitoring"

**Trading Page Layout:**
```
┌─ Trading ──────────────────────────────────────────────────┐
│ ┌─ Account Status ──────────────────────────────────────┐ │
│ │ [badge: Connected/Disconnected]  Buying Power: $xxx   │ │
│ │ Portfolio Value: $xxx                                  │ │
│ └───────────────────────────────────────────────────────┘ │
│ ┌─ Current Positions ───────────────────────────────────┐ │
│ │ Symbol | Qty | Avg Price | Current | P&L              │ │
│ └───────────────────────────────────────────────────────┘ │
│ ┌─ Pending Orders ──────────────────────────────────────┐ │
│ │ Date: [dropdown]           [Execute Orders] button    │ │
│ │ Symbol | Direction | Target Weight | Est. Value       │ │
│ └───────────────────────────────────────────────────────┘ │
│ ┌─ Trade History ───────────────────────────────────────┐ │
│ │ Symbol | Side | Qty | Price | Time                   │ │
│ └───────────────────────────────────────────────────────┘ │
│ ┌─ Settings ────────────────────────────────────────────┐ │
│ │ [x] Auto-execute on pipeline completion               │ │
│ │ Max daily orders: [20]  |  Turnover limit: [30%]    │ │
│ └───────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────┘
```

---

## 4. Safety Guards

| Check | Limit | Behavior on Violation |
|-------|-------|----------------------|
| Max orders/day | 20 | Block execution, show warning |
| Max turnover | 30% | Block execution, show warning |
| Max position size | 10% of portfolio | Skip/reject individual order |
| Paper mode only | `paper-api.alpaca.markets` in URL | Block live trading |
| Buying power | > 0 | Block if account has no funds |

---

## 5. Configuration

### `config/runtime.env` additions

```env
MLCOUNCIL_AUTO_EXECUTE=false
MLCOUNCIL_MAX_DAILY_ORDERS=20
MLCOUNCIL_MAX_TURNOVER=0.30
MLCOUNCIL_MAX_POSITION_SIZE=0.10
```

---

## 6. File Changes

| File | Action |
|------|--------|
| `api/routers/trading.py` | New — trading endpoints |
| `api/services/trading_service.py` | New — execution logic |
| `api/main.py` | Add `include_router(trading.router)` |
| `api/templates/admin.html` | Add Trading nav link + page HTML |
| `api/static/js/admin.js` | Add trading UI functions |
| `api/static/css/admin.css` | Add trading page styles |
| `config/runtime.env` | Add trading config vars |
| `AGENTS.md` | Document new service |

---

## 7. Error Handling

- Alpaca API failures → Return error details, log locally, do not crash
- No pending orders → Show "No orders found for this date"
- Already executed today → Show "Orders already executed for {date}"
- Network timeout → Retry once, then return error

---

## 8. Out of Scope

- Live trading (non-paper) — blocked by safety check
- Order modification/cancellation UI
- Partial fill handling in UI
- WebSocket real-time updates
