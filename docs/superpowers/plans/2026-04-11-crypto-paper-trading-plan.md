# Crypto Paper Trading via Alpaca — Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development or superpowers:executing-plans to implement. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend MLCouncil to support BTC/USD and ETH/USD paper trading on Alpaca alongside existing equity flow.

**Approach:** Alpaca's crypto API uses the same order submission model as equities. The main changes are: add crypto data ingestion, expand the universe config, and update the adapter to route crypto orders to the correct endpoint.

---

## Chunk 1: Adapter Updates

### Files
- Modify: `execution/alpaca_adapter.py`

### Tasks

- [ ] **Step 1: Detect asset class in `submit_order`**

Alpaca distinguishes equity and crypto orders by symbol format. Add a helper:

```python
def _is_crypto(symbol: str) -> bool:
    return "/" in symbol or symbol.upper() in ("BTCUSD", "ETHUSD", "BTC/USD", "ETH/USD")

def submit_order(self, symbol: str, qty: int, side: str, order_type: str = "market") -> dict:
    if self._is_crypto(symbol):
        return self._submit_crypto_order(symbol, qty, side, order_type)
    return self._submit_equity_order(symbol, qty, side, order_type)
```

- [ ] **Step 2: Add `_submit_crypto_order`**

Alpaca Crypto API endpoint is `https://api.alpaca.markets/v2/orders`. Same payload format as equity — `symbol`, `qty`, `side`, `type`. No `notional` required for market orders.

```python
def _submit_crypto_order(self, symbol: str, qty: int, side: str, order_type: str) -> dict:
    payload = {
        "symbol": symbol.upper().replace("/", ""),  # "BTCUSD" not "BTC/USD"
        "qty": str(qty),
        "side": side.lower(),
        "type": order_type,
        "time_in_force": "day",
    }
    return self._post("/v2/orders", payload)
```

- [ ] **Step 3: Add `_get_crypto_positions`**

```python
def get_crypto_positions(self) -> pd.DataFrame:
    resp = self._get("/v2/positions?asset_class=crypto")
    return pd.DataFrame(resp)
```

- [ ] **Step 4: Update `get_all_positions` to include crypto**

```python
def get_all_positions(self) -> pd.DataFrame:
    equity = self._get_equity_positions()
    crypto = self.get_crypto_positions()
    return pd.concat([equity, crypto], ignore_index=True)
```

---

## Chunk 2: Data Ingestion for Crypto

### Files
- Modify: `data/ingest/market_data.py` (or create `data/ingest/crypto_data.py`)
- Modify: `config/universe.yaml`

### Tasks

- [ ] **Step 1: Add crypto tickers to universe**

```yaml
crypto_universe:
  large_cap:
    - BTCUSD
    - ETHUSD
```

- [ ] **Step 2: Fetch crypto OHLCV from Alpaca Data**

Alpaca provides crypto bars via `GET /v2/stocks/{symbol}/bars` (same endpoint, crypto symbols work). Or use `yfinance` for BTC-USD, ETH-USD.

```python
def fetch_crypto_bars(symbol: str, start: str, end: str, timeframe: str = "1Day") -> pd.DataFrame:
    # yfinance handles crypto natively
    ticker = yf.Ticker(symbol.replace("USD", "-USD"))
    df = ticker.history(start=start, end=end, interval="1d")
    df["symbol"] = symbol
    return df.reset_index()
```

- [ ] **Step 3: Add `fetch_crypto_data` to pipeline ingest**

```python
# In data/ingest/market_data.py or new module
def ingest_crypto(symbols: list[str], start: date, end: date) -> pd.DataFrame:
    frames = [fetch_crypto_bars(s, start, end) for s in symbols]
    return pd.concat(frames, ignore_index=True)
```

---

## Chunk 3: Portfolio and Risk Extensions

### Files
- Modify: `council/portfolio.py` (handle crypto in optimizer)
- Modify: `council/risk_engine.py` (crypto position limits)

### Tasks

- [ ] **Step 1: Add crypto-specific position limits**

In `config/runtime.env`:

```env
MLCOUNCIL_CRYPTO_ENABLED=true
MLCOUNCIL_MAX_CRYPTO_POSITION_SIZE=0.20   # higher limit for crypto vs 10% equity
MLCOUNCIL_MAX_CRYPTO_TURNOVER=0.40
```

- [ ] **Step 2: Portfolio optimizer — skip crypto if disabled**

```python
def optimize_portfolio(self, signals, portfolio_value, enabled_crypto=True):
    equity_signals = signals[~signals["symbol"].isin(CRYPTO_TICKERS)]
    crypto_signals = signals[signals["symbol"].isin(CRYPTO_TICKERS)] if enabled_crypto else pd.DataFrame()

    equity_weights = self._optimize_group(equity_signals, ...)
    crypto_weights = self._optimize_group(crypto_signals, ...) if enabled_crypto else {}

    return {**equity_weights, **crypto_weights}
```

- [ ] **Step 3: Risk engine — different max drawdown for crypto**

Crypto is more volatile. Add regime-conditional limits in `risk_engine.py`:

```python
def check_risk_limits(positions: pd.DataFrame, is_crypto: bool = False) -> tuple[bool, str]:
    max_position = float(os.getenv("MLCOUNCIL_MAX_CRYPTO_POSITION_SIZE", "0.20")) if is_crypto \
                   else float(os.getenv("MLCOUNCIL_MAX_POSITION_SIZE", "0.10"))
    ...
```

---

## Chunk 4: Trading Service and UI

### Files
- Modify: `api/services/trading_service.py`
- Modify: `api/routers/trading.py` (optional: add `?asset_class=crypto` filter)
- Modify: `dashboard/app.py` (show crypto positions)

### Tasks

- [ ] **Step 1: Trading service — route to correct adapter method**

`trading_service.py` already calls `node.get_all_positions()` and `node.submit_order()`. No changes needed if adapter handles routing internally.

- [ ] **Step 2: Add crypto to dashboard positions table**

In `dashboard/app.py`, add a toggle or filter for "Show crypto positions":

```python
st.subheader("Crypto Positions")
crypto_pos = [p for p in positions if p["symbol"] in CRYPTO_TICKERS]
if crypto_pos:
    st.dataframe(crypto_pos)
```

---

## Chunk 5: Testing

### Files
- Create: `tests/test_crypto_adapter.py`

### Tasks

- [ ] **Step 1: Test crypto symbol detection**

```python
def test_is_crypto():
    assert AlpacaLiveNode._is_crypto("BTCUSD") is True
    assert AlpacaLiveNode._is_crypto("ETH/USD") is True
    assert AlpacaLiveNode._is_crypto("AAPL") is False
```

- [ ] **Step 2: Test order routing**

Mock `_post` and verify BTC order goes to `/v2/orders` with `symbol=BTCUSD`.

- [ ] **Step 3: Integration test with paper account**

Requires `ALPACA_API_KEY` with crypto permissions. Skip if not configured.

---

## Verification

1. `python -c "from execution.alpaca_adapter import AlpacaLiveNode; print(AlpacaLiveNode._is_crypto('BTCUSD'))"` → `True`
2. `python -m pytest tests/test_crypto_adapter.py -v`
3. Visit Alpaca dashboard — confirm BTC-USD and ETH-USD appear in paper account
4. Run full pipeline with crypto in universe, verify orders written to parquet
5. Execute via Admin UI, verify orders appear in Alpaca paper account

## Dependencies

- No new packages needed — Alpaca SDK supports crypto, yfinance handles crypto OHLCV
- Alpaca paper account must have crypto trading enabled (usually on by default)
