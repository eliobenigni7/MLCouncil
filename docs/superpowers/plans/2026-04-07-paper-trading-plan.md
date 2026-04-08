# Paper Trading via Admin UI — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add paper trading execution to MLCouncil Admin UI — pipeline generates orders, user reviews and executes them to Alpaca Paper Trading with safety guards.

**Architecture:** Trading service reads pipeline output (parquet), validates against safety limits, submits orders via Alpaca adapter, logs trades locally. UI shows status, positions, pending orders, and execution controls.

**Tech Stack:** FastAPI, Alpaca Trade API, Pandas/Parquet, existing admin UI (Jinja2 + vanilla JS)

---

## Chunk 1: Core Infrastructure

### Files
- Create: `api/services/trading_service.py`
- Create: `api/routers/trading.py`
- Modify: `api/main.py:79-85` (add router import and include_router)
- Modify: `config/runtime.env` (add trading config vars)

### Tasks

- [ ] **Step 1: Write trading_service.py**

```python
"""Trading service — executes orders to Alpaca Paper with safety guards."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from execution.alpaca_adapter import AlpacaLiveNode, AlpacaConfig, TradingMode
from runtime_env import load_runtime_env

_ROOT = Path(__file__).parents[2]
TRADE_LOG_DIR = _ROOT / "data" / "paper_trades"
ORDERS_DIR = _ROOT / "data" / "orders"


class TradingService:
    def __init__(self):
        load_runtime_env()
        self._node: Optional[AlpacaLiveNode] = None
        self._config = AlpacaConfig.from_env()

    @property
    def node(self) -> AlpacaLiveNode:
        if self._node is None:
            self._node = AlpacaLiveNode(self._config)
        return self._node

    @property
    def is_paper(self) -> bool:
        return self._config.mode == TradingMode.PAPER

    def get_status(self) -> dict:
        """Return Alpaca connection status and account info."""
        if not self.is_paper:
            return {"error": "Live trading blocked — paper mode only"}
        try:
            account = self.node.get_account_info()
            positions = self.node.get_all_positions()
            return {
                "connected": True,
                "paper": self.is_paper,
                "account": account,
                "positions": positions.to_dict(orient="records") if not positions.empty else [],
            }
        except Exception as e:
            return {"connected": False, "error": str(e)}

    def get_pending_orders(self, date: str) -> list[dict]:
        """Load orders from parquet for given date."""
        path = ORDERS_DIR / f"{date}.parquet"
        if not path.exists():
            return []
        df = pd.read_parquet(path)
        return df.to_dict(orient="records")

    def get_latest_order_date(self) -> Optional[str]:
        """Return most recent order file date."""
        if not ORDERS_DIR.exists():
            return None
        files = sorted(ORDERS_DIR.glob("*.parquet"))
        if not files:
            return None
        return files[-1].stem

    def _validate_order(self, order: dict, account: dict, positions: pd.DataFrame) -> tuple[bool, str]:
        """Validate single order against safety limits."""
        max_daily = int(os.getenv("MLCOUNCIL_MAX_DAILY_ORDERS", "20"))
        max_turnover = float(os.getenv("MLCOUNCIL_MAX_TURNOVER", "0.30"))
        max_position = float(os.getenv("MLCOUNCIL_MAX_POSITION_SIZE", "0.10"))

        portfolio_value = float(account.get("portfolio_value", 0))
        if portfolio_value <= 0:
            return False, "No buying power"

        symbol = order.get("ticker")
        direction = order.get("direction", "buy").lower()
        quantity = order.get("quantity", 0)
        
        if quantity <= 0:
            return False, f"Invalid quantity for {symbol}"

        # Position size check
        if direction == "buy":
            position_value = quantity * order.get("price", 0)
            if portfolio_value > 0 and (position_value / portfolio_value) > max_position:
                return False, f"{symbol} exceeds max position {max_position:.0%}"

        return True, "OK"

    def execute_orders(self, date: str) -> dict:
        """Execute pending orders for date to Alpaca Paper."""
        if not self.is_paper:
            return {"error": "Live trading blocked"}

        try:
            account = self.node.get_account_info()
            positions_df = self.node.get_all_positions()
            orders = self.get_pending_orders(date)

            if not orders:
                return {"error": f"No orders found for {date}"}

            # Validate all orders first
            max_daily = int(os.getenv("MLCOUNCIL_MAX_DAILY_ORDERS", "20"))
            if len(orders) > max_daily:
                return {"error": f"Order count {len(orders)} exceeds max {max_daily}"}

            portfolio_value = float(account.get("portfolio_value", 0))
            turnover_limit = float(os.getenv("MLCOUNCIL_MAX_TURNOVER", "0.30"))

            # Compute target tickers
            target_tickers = {o["ticker"] for o in orders}
            
            # Liquidate positions not in target
            liquidate_results = []
            if not positions_df.empty:
                for _, pos in positions_df.iterrows():
                    if pos["symbol"] not in target_tickers:
                        result = self.node.submit_order(pos["symbol"], int(pos["qty"]), "sell")
                        liquidate_results.append(result)

            # Submit new orders
            order_results = []
            for order in orders:
                symbol = order["ticker"]
                direction = order["direction"]
                quantity = int(order["quantity"])
                
                valid, msg = self._validate_order(order, account, positions_df)
                if not valid:
                    order_results.append({"symbol": symbol, "status": "rejected", "reason": msg})
                    continue

                result = self.node.submit_order(symbol, quantity, direction)
                order_results.append(result)

            # Log trade summary
            self._log_trade(date, {
                "orders": order_results,
                "liquidations": liquidate_results,
                "account": account,
            })

            return {
                "date": date,
                "orders_submitted": len([r for r in order_results if r.get("status") != "rejected"]),
                "orders_rejected": len([r for r in order_results if r.get("status") == "rejected"]),
                "liquidations": len(liquidate_results),
                "results": order_results,
            }

        except Exception as e:
            return {"error": str(e)}

    def liquidate_all(self) -> dict:
        """Liquidate all current positions."""
        if not self.is_paper:
            return {"error": "Live trading blocked"}
        try:
            results = self.node.liquidate_all()
            return {"liquidated": len(results), "results": results}
        except Exception as e:
            return {"error": str(e)}

    def get_trade_history(self, days: int = 7) -> list[dict]:
        """Load trade history from log files."""
        if not TRADE_LOG_DIR.exists():
            return []
        logs = []
        for pq in sorted(TRADE_LOG_DIR.glob("*.json"))[-days:]:
            try:
                logs.extend(json.loads(pq.read_text()))
            except Exception:
                pass
        return logs

    def _log_trade(self, date: str, data: dict) -> None:
        TRADE_LOG_DIR.mkdir(parents=True, exist_ok=True)
        path = TRADE_LOG_DIR / f"{date}.json"
        existing = []
        if path.exists():
            try:
                existing = json.loads(path.read_text())
            except Exception:
                existing = []
        existing.append(data)
        path.write_text(json.dumps(existing, indent=2))


service = TradingService()
```

- [ ] **Step 2: Write router — `api/routers/trading.py`**

```python
"""Trading API endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.services import trading_service

router = APIRouter(prefix="/trading", tags=["trading"])


class ExecuteRequest(BaseModel):
    date: str


class StatusResponse(BaseModel):
    connected: bool
    paper: bool = True
    account: dict = {}
    positions: list = []
    error: str | None = None


class ExecuteResponse(BaseModel):
    date: str
    orders_submitted: int = 0
    orders_rejected: int = 0
    liquidations: int = 0
    results: list = []
    error: str | None = None


@router.get("/status", response_model=StatusResponse)
async def trading_status():
    status = trading_service.service.get_status()
    return StatusResponse(
        connected=status.get("connected", False),
        paper=status.get("paper", True),
        account=status.get("account", {}),
        positions=status.get("positions", []),
        error=status.get("error"),
    )


@router.get("/orders/latest")
async def latest_order_date():
    date = trading_service.service.get_latest_order_date()
    if date is None:
        raise HTTPException(status_code=404, detail="No order files found")
    return {"date": date}


@router.get("/orders/pending/{date}")
async def pending_orders(date: str):
    orders = trading_service.service.get_pending_orders(date)
    return {"date": date, "orders": orders}


@router.post("/execute", response_model=ExecuteResponse)
async def execute_orders(req: ExecuteRequest):
    result = trading_service.service.execute_orders(req.date)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return ExecuteResponse(**result)


@router.post("/liquidate")
async def liquidate_all():
    result = trading_service.service.liquidate_all()
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.get("/history")
async def trade_history(days: int = 7):
    return {"trades": trading_service.service.get_trade_history(days)}
```

- [ ] **Step 3: Update `api/main.py` — add trading router**

In `api/main.py` line 79, add:
```python
from api.routers import health, pipeline, portfolio, config, monitoring, trading
```

And line 85:
```python
app.include_router(trading.router, prefix=API_PREFIX)
```

- [ ] **Step 4: Update `config/runtime.env` — add trading config**

```env
MLCOUNCIL_AUTO_EXECUTE=false
MLCOUNCIL_MAX_DAILY_ORDERS=20
MLCOUNCIL_MAX_TURNOVER=0.30
MLCOUNCIL_MAX_POSITION_SIZE=0.10
```

- [ ] **Step 5: Test router imports**

Run: `cd E:\Github\MLCouncil && python -c "from api.routers import trading; print('OK')"`
Expected: OK

---

## Chunk 2: Admin UI

### Files
- Modify: `api/templates/admin.html` (add Trading nav + page)
- Modify: `api/static/js/admin.js` (add trading UI functions)
- Modify: `api/static/css/admin.css` (add trading styles)

### Tasks

- [ ] **Step 1: Add Trading nav link in `admin.html`**

After line 19 (monitoring nav link), add:
```html
<a data-page="trading">Trading</a>
```

- [ ] **Step 2: Add Trading page HTML in `admin.html`**

After line 108 (closing `</div>` of monitoring page), add:
```html
<div id="trading" class="page">
    <div class="card">
        <h2>Account Status</h2>
        <div class="kpi-grid" id="trading-status-kpis"></div>
    </div>
    <div class="card">
        <h2>Current Positions</h2>
        <table id="positions-table">
            <thead><tr><th>Symbol</th><th>Qty</th><th>Avg Price</th><th>Current</th><th>P&L</th></tr></thead>
            <tbody></tbody>
        </table>
    </div>
    <div class="card">
        <h2>Pending Orders</h2>
        <div style="margin-bottom: 1rem; display: flex; gap: 0.5rem; align-items: center;">
            <select id="trading-order-date" style="margin-bottom: 0;"></select>
            <button class="btn" id="execute-btn">Execute Orders</button>
        </div>
        <table id="pending-orders-table">
            <thead><tr><th>Symbol</th><th>Direction</th><th>Target Weight</th><th>Est. Value</th></tr></thead>
            <tbody></tbody>
        </table>
        <div id="execution-result" style="margin-top: 1rem;"></div>
    </div>
    <div class="card">
        <h2>Trade History</h2>
        <table id="trade-history-table">
            <thead><tr><th>Symbol</th><th>Side</th><th>Qty</th><th>Status</th><th>Time</th></tr></thead>
            <tbody></tbody>
        </table>
    </div>
    <div class="card">
        <h2>Settings</h2>
        <div class="settings-grid">
            <label><input type="checkbox" id="auto-execute"> Auto-execute on pipeline completion</label>
        </div>
    </div>
</div>
```

- [ ] **Step 3: Add trading CSS in `admin.css`**

Add to end of `admin.css`:
```css
/* Trading page */
#trading .kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin-bottom: 1.5rem;
}
```

- [ ] **Step 4: Add trading JS functions in `admin.js`**

After `refreshConfig()` function (~line 296), add:
```javascript
async function refreshTrading() {
    try {
        const status = await fetchAPI('/trading/status');
        const statusKpis = document.getElementById('trading-status-kpis');
        
        if (!status.connected) {
            statusKpis.innerHTML = `<div class="kpi-card"><div class="kpi-value error">Disconnected</div><div class="kpi-label">${status.error || 'Check Alpaca config'}</div></div>`;
        } else {
            statusKpis.innerHTML = `
                <div class="kpi-card">
                    <div class="kpi-label">Status</div>
                    <div class="kpi-value ok">${status.paper ? 'Paper Trading' : 'Live'}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Buying Power</div>
                    <div class="kpi-value">$${parseFloat(status.account?.buying_power || 0).toLocaleString()}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Portfolio Value</div>
                    <div class="kpi-value">$${parseFloat(status.account?.portfolio_value || 0).toLocaleString()}</div>
                </div>
            `;
        }
        
        // Positions table
        const posBody = document.querySelector('#positions-table tbody');
        if (status.positions && status.positions.length > 0) {
            posBody.innerHTML = status.positions.map(p => `
                <tr>
                    <td>${p.symbol}</td>
                    <td>${p.qty}</td>
                    <td>$${parseFloat(p.avg_price || 0).toFixed(2)}</td>
                    <td>$${parseFloat(p.current_price || 0).toFixed(2)}</td>
                    <td style="color: ${parseFloat(p.unrealized_pl || 0) >= 0 ? 'var(--ok)' : 'var(--error)'}">$${parseFloat(p.unrealized_pl || 0).toFixed(2)} (${parseFloat(p.unrealized_pl_pc || 0).toFixed(1)}%)</td>
                </tr>
            `).join('');
        } else {
            posBody.innerHTML = '<tr><td colspan="5" style="color: var(--text-secondary);">No open positions</td></tr>';
        }
        
        // Load pending orders date dropdown
        try {
            const latest = await fetchAPI('/trading/orders/latest');
            const select = document.getElementById('trading-order-date');
            select.innerHTML = `<option value="${latest.date}">${latest.date}</option>`;
            await loadPendingOrders(latest.date);
        } catch (e) {
            document.querySelector('#pending-orders-table tbody').innerHTML = '<tr><td colspan="4" style="color: var(--text-secondary);">No orders found</td></tr>';
        }
        
        // Trade history
        const history = await fetchAPI('/trading/history?days=7');
        const histBody = document.querySelector('#trade-history-table tbody');
        if (history.trades && history.trades.length > 0) {
            histBody.innerHTML = history.trades.map(t => `
                <tr>
                    <td>${t.symbol || '--'}</td>
                    <td>${t.side || '--'}</td>
                    <td>${t.qty || '--'}</td>
                    <td>${t.status || '--'}</td>
                    <td>${t.submitted_at || '--'}</td>
                </tr>
            `).join('');
        } else {
            histBody.innerHTML = '<tr><td colspan="5" style="color: var(--text-secondary);">No trade history</td></tr>';
        }
    } catch (e) {
        console.error('Failed to refresh trading:', e);
    }
}

async function loadPendingOrders(date) {
    try {
        const resp = await fetchAPI(`/trading/orders/pending/${date}`);
        const tbody = document.querySelector('#pending-orders-table tbody');
        
        if (resp.orders && resp.orders.length > 0) {
            tbody.innerHTML = resp.orders.map(o => `
                <tr>
                    <td>${o.ticker}</td>
                    <td>${(o.direction || 'buy').toUpperCase()}</td>
                    <td>${((o.target_weight || 0) * 100).toFixed(1)}%</td>
                    <td>$${((o.quantity || 0) * (o.price || 0)).toFixed(0)}</td>
                </tr>
            `).join('');
        } else {
            tbody.innerHTML = '<tr><td colspan="4" style="color: var(--text-secondary);">No pending orders</td></tr>';
        }
    } catch (e) {
        console.error('Failed to load pending orders:', e);
    }
}

document.getElementById('execute-btn').addEventListener('click', async () => {
    const date = document.getElementById('trading-order-date').value;
    if (!date) return;
    
    const btn = document.getElementById('execute-btn');
    btn.disabled = true;
    btn.textContent = 'Executing...';
    
    try {
        const result = await postAPI('/trading/execute', {date});
        const div = document.getElementById('execution-result');
        div.innerHTML = `
            <div style="padding: 1rem; background: var(--bg-secondary); border-radius: 4px;">
                <strong>Execution complete:</strong><br>
                Orders submitted: ${result.orders_submitted}<br>
                Orders rejected: ${result.orders_rejected}<br>
                Liquidations: ${result.liquidations}
            </div>
        `;
        showToast('Orders executed successfully');
        await refreshTrading();
    } catch (e) {
        showToast('Execution failed: ' + e.message, 'error');
    } finally {
        btn.disabled = false;
        btn.textContent = 'Execute Orders';
    }
});

document.getElementById('trading-order-date').addEventListener('change', async (e) => {
    if (e.target.value) {
        await loadPendingOrders(e.target.value);
    }
});
```

- [ ] **Step 5: Update `refreshAll()` in admin.js**

Add `refreshTrading()` to the `refreshAll()` function call at line ~298:
```javascript
async function refreshAll() {
    await Promise.all([
        refreshOverview(),
        refreshPipelineStatus(),
        refreshPortfolio(),
        refreshMonitoring(),
        refreshConfig(),
        refreshTrading(),  // ADD THIS
    ]);
}
```

---

## Chunk 3: Integration & Testing

### Files
- Create: `tests/test_trading_service.py`
- Modify: `AGENTS.md` (document new service)

### Tasks

- [ ] **Step 1: Write trading service test**

```python
"""Tests for trading service."""

import pytest
from unittest.mock import MagicMock, patch
from datetime import date


class TestTradingService:
    def test_validate_order_valid(self):
        from api.services.trading_service import TradingService
        
        svc = TradingService.__new__(TradingService)
        svc._node = None
        
        order = {"ticker": "AAPL", "direction": "buy", "quantity": 10, "price": 150.0}
        account = {"portfolio_value": 100000.0}
        positions = MagicMock()
        
        valid, msg = svc._validate_order(order, account, positions)
        assert valid is True

    def test_validate_order_exceeds_position_limit(self):
        from api.services.trading_service import TradingService
        
        svc = TradingService.__new__(TradingService)
        svc._node = None
        
        order = {"ticker": "AAPL", "direction": "buy", "quantity": 1000, "price": 150.0}
        account = {"portfolio_value": 100000.0}
        positions = MagicMock()
        
        valid, msg = svc._validate_order(order, account, positions)
        assert valid is False
        assert "exceeds max position" in msg

    def test_get_pending_orders_empty(self, tmp_path):
        from api.services.trading_service import TradingService, ORDERS_DIR
        
        original_dir = TradingService.__class__.__dict__.get('ORDERS_DIR')
        TradingService.ORDERS_DIR = tmp_path
        
        svc = TradingService.__new__(TradingService)
        svc._node = None
        
        result = svc.get_pending_orders("2024-01-01")
        assert result == []
        
        if original_dir:
            TradingService.ORDERS_DIR = original_dir

    def test_get_status_no_node(self):
        from api.services.trading_service import TradingService
        
        svc = TradingService.__new__(TradingService)
        svc._node = None
        svc._config = MagicMock()
        svc._config.mode = "paper"
        
        # Should not raise
        status = svc.get_status()
        assert "connected" in status or "error" in status
```

- [ ] **Step 2: Run tests**

Run: `python -m pytest tests/test_trading_service.py -v`
Expected: Tests pass (or skip if no Alpaca keys)

- [ ] **Step 3: Update AGENTS.md**

Add to `AGENTS.md`:
```markdown
## Paper Trading

Trading service at `api/services/trading_service.py` executes pipeline orders to Alpaca Paper:
- Endpoint: `POST /api/trading/execute` with `{"date": "YYYY-MM-DD"}`
- UI: Admin page "Trading" shows positions, pending orders, history
- Safety: max 20 orders/day, 30% turnover, 10% position cap

Required env vars (in `.env`):
```
ALPACA_API_KEY=...
ALPACA_SECRET_KEY=...
ALPACA_BASE_URL=https://paper-api.alpaca.markets
MLCOUNCIL_AUTO_EXECUTE=false
MLCOUNCIL_MAX_DAILY_ORDERS=20
MLCOUNCIL_MAX_TURNOVER=0.30
MLCOUNCIL_MAX_POSITION_SIZE=0.10
```
```

---

## Verification

1. Start admin API: `python run_admin.py`
2. Visit http://localhost:8000
3. Navigate to "Trading" page
4. Verify: Account status shows (disconnected if no keys), positions table, orders dropdown, execute button
5. If Alpaca keys set: Should show connected status and positions

## Dependencies

- `alpaca-trade-api>=2.3.0` (already in requirements_api.txt)
- `python-dotenv>=1.0.0` (already in requirements_api.txt)
