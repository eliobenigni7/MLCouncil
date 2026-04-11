"""Alpaca execution adapter for paper and live trading.

Provides a bridge between MLCouncil's order generation and Alpaca's API.
Supports both paper trading (sandbox) and live trading modes.

Usage:
    from execution.alpaca_adapter import AlpacaLiveNode, AlpacaConfig

    config = AlpacaConfig(
        paper_key="YOUR_PAPER_KEY",
        paper_secret="YOUR_PAPER_SECRET",
        mode="paper"
    )
    node = AlpacaLiveNode(config)
    node.submit_order("AAPL", 100, "buy")
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

_TRADE_LOG_LOCK = threading.Lock()

import pandas as pd
from runtime_env import load_runtime_env

load_runtime_env()

_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(_ROOT))
logger = logging.getLogger(__name__)


class TradingMode(Enum):
    PAPER = "paper"
    LIVE = "live"


@dataclass
class AlpacaConfig:
    paper_key: str
    paper_secret: str
    live_key: Optional[str] = None
    live_secret: Optional[str] = None
    mode: TradingMode = TradingMode.PAPER

    @classmethod
    def from_env(cls) -> "AlpacaConfig":
        mode = TradingMode(os.getenv("TRADING_MODE", "paper"))
        return cls(
            paper_key=os.getenv("ALPACA_PAPER_KEY", "") or os.getenv("ALPACA_API_KEY", ""),
            paper_secret=os.getenv("ALPACA_PAPER_SECRET", "") or os.getenv("ALPACA_SECRET_KEY", ""),
            live_key=os.getenv("ALPACA_LIVE_KEY"),
            live_secret=os.getenv("ALPACA_LIVE_SECRET"),
            mode=mode,
        )

    def validate(self) -> None:
        if self.mode == TradingMode.PAPER:
            if not self.paper_key or not self.paper_secret:
                raise EnvironmentError(
                    "Alpaca paper trading requires ALPACA_PAPER_KEY and "
                    "ALPACA_PAPER_SECRET environment variables."
                )
        elif self.mode == TradingMode.LIVE:
            if not self.live_key or not self.live_secret:
                raise EnvironmentError(
                    "Alpaca live trading requires ALPACA_LIVE_KEY and "
                    "ALPACA_LIVE_SECRET environment variables."
                )


class AlpacaLiveNode:
    def __init__(self, config: AlpacaConfig):
        self.config = config
        self.config.validate()

        self._trading_client = None
        self._data_client = None
        self._trade_log_dir = _ROOT / "data" / "paper_trades"
        self._trade_log_dir.mkdir(parents=True, exist_ok=True)

        self._initialize_api()

    def _initialize_api(self) -> None:
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.data.historical.stock import StockHistoricalDataClient
        except ImportError:
            raise ImportError(
                "alpaca-py is required for live trading. "
                "Install with: pip install alpaca"
            )

        if self.config.mode == TradingMode.PAPER:
            key = self.config.paper_key
            secret = self.config.paper_secret
            paper = True
        else:
            key = self.config.live_key
            secret = self.config.live_secret
            paper = False

        self._trading_client = TradingClient(key, secret, paper=paper)
        self._data_client = StockHistoricalDataClient(key, secret)

    @property
    def is_paper(self) -> bool:
        return self.config.mode == TradingMode.PAPER

    def is_tradeable(self, symbol: str) -> bool:
        try:
            from alpaca.trading.requests import GetAssetRequest
            from alpaca.trading.enums import AssetClass

            request = GetAssetRequest(symbol=symbol)
            asset = self._trading_client.get_asset(request)
            return asset.tradable and asset.fractionable
        except Exception:
            return False

    def get_position(self, symbol: str) -> Optional[dict]:
        try:
            from alpaca.trading.requests import GetPositionRequest

            request = GetPositionRequest(symbol=symbol)
            pos = self._trading_client.get_open_position(request)
            return {
                "symbol": symbol,
                "qty": round(float(pos.qty), 2),
                "avg_price": float(pos.avg_entry_price),
                "current_value": float(pos.market_value),
                "current_price": float(pos.current_price) if pos.current_price else float(pos.avg_entry_price),
            }
        except Exception:
            return None

    def get_latest_price(self, symbol: str) -> Optional[float]:
        try:
            from alpaca.data.requests import StockLatestTradeRequest

            request = StockLatestTradeRequest(symbol_or_symbols=symbol)
            trades = self._data_client.get_stock_latest_trade(request)
            if symbol in trades:
                return float(trades[symbol].price)
            return None
        except Exception:
            return None

    def get_crypto_positions(self) -> pd.DataFrame:
        """Get crypto positions via direct HTTP API."""
        import requests

        api_key = self.config.paper_key or self.config.live_key
        api_secret = self.config.paper_secret or self.config.live_secret
        base_url = "https://paper-api.alpaca.markets" if self.config.mode == TradingMode.PAPER else "https://api.alpaca.markets"

        headers = {
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": api_secret,
        }

        try:
            resp = requests.get(f"{base_url}/v2/positions?asset_class=crypto", headers=headers, timeout=30)
            if not resp.ok:
                logger.warning(f"Failed to fetch crypto positions: {resp.status_code}")
                return pd.DataFrame()

            positions = resp.json()
            if not positions:
                return pd.DataFrame()

            def _float(val, default=0.0):
                try:
                    return float(val)
                except (TypeError, ValueError):
                    return default

            return pd.DataFrame([
                {
                    "symbol": p.get("symbol", ""),
                    "qty": round(float(p.get("qty", 0)), 2),
                    "avg_price": _float(p.get("avg_entry_price")),
                    "current_price": _float(p.get("current_price")),
                    "current_value": _float(p.get("market_value")),
                    "unrealized_pnl": _float(p.get("unrealized_pl")),
                    "unrealized_pnl_pct": _float(p.get("unrealized_plpc")),
                    "asset_class": "crypto",
                }
                for p in positions
            ])
        except Exception as e:
            logger.warning(f"Error fetching crypto positions: {e}")
            return pd.DataFrame()

    def get_all_positions(self, strict: bool = False) -> pd.DataFrame:
        equity_df = pd.DataFrame()
        try:
            positions = self._trading_client.get_all_positions()
            if positions:
                def _position_float(position, *names: str, default: float = 0.0) -> float:
                    for name in names:
                        value = getattr(position, name, None)
                        if value is not None:
                            return float(value)
                    return default

                equity_df = pd.DataFrame([
                    {
                        "symbol": p.symbol,
                        "qty": round(float(p.qty), 2),
                        "avg_price": _position_float(p, "avg_entry_price"),
                        "current_price": _position_float(p, "current_price"),
                        "current_value": _position_float(p, "market_value"),
                        "unrealized_pnl": _position_float(p, "unrealized_pl"),
                        "unrealized_pnl_pct": _position_float(p, "unrealized_plpc"),
                        "asset_class": "equity",
                    }
                    for p in positions
                ])
        except Exception as e:
            if strict:
                raise RuntimeError(f"Error loading Alpaca equity positions: {e}") from e

        crypto_df = pd.DataFrame()
        try:
            crypto_df = self.get_crypto_positions()
        except Exception:
            pass

        if equity_df.empty and crypto_df.empty:
            return pd.DataFrame()
        elif equity_df.empty:
            return crypto_df
        elif crypto_df.empty:
            return equity_df
        return pd.concat([equity_df, crypto_df], ignore_index=True)

    def get_account_info(self) -> dict:
        account = self._trading_client.get_account()
        return {
            "buying_power": float(account.buying_power),
            "cash": float(account.cash),
            "portfolio_value": float(account.portfolio_value),
            "status": account.status,
            "mode": self.config.mode.value,
        }

    @staticmethod
    def _is_crypto(symbol: str) -> bool:
        """Return True if symbol represents a crypto asset."""
        upper = symbol.upper()
        return (
            "/" in symbol
            or upper in ("BTCUSD", "ETHUSD", "BTC/USD", "ETH/USD")
            or upper.endswith("USD")
            and upper not in ("AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA")
            and not upper.isalpha()
        )

    def _submit_crypto_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str = "market",
        time_in_force: str = "day",
        limit_price: Optional[float] = None,
    ) -> dict:
        """Submit a crypto order via direct HTTP (alpaca-py doesn't support crypto orders)."""
        import requests

        api_key = self.config.paper_key or self.config.live_key
        api_secret = self.config.paper_secret or self.config.live_secret
        base_url = "https://paper-api.alpaca.markets" if self.config.mode == TradingMode.PAPER else "https://api.alpaca.markets"

        headers = {
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": api_secret,
            "Content-Type": "application/json",
        }

        # Normalize symbol: BTCUSD not BTC-USD or BTC/USD
        normalized = symbol.upper().replace("/", "").replace("-", "")
        if not normalized.endswith("USD"):
            normalized = normalized + "USD"

        payload = {
            "symbol": normalized,
            "qty": str(qty),
            "side": side.lower(),
            "type": order_type.lower(),
            "time_in_force": time_in_force.lower(),
        }
        if limit_price is not None:
            payload["limit_price"] = str(limit_price)

        resp = requests.post(f"{base_url}/v2/orders", json=payload, headers=headers, timeout=30)
        if not resp.ok:
            raise RuntimeError(f"Crypto order failed: {resp.status_code} {resp.text}")

        order = resp.json()
        trade_record = {
            "order_id": order.get("id", ""),
            "symbol": normalized,
            "qty": qty,
            "side": side,
            "order_type": order_type,
            "status": order.get("status", "unknown"),
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "mode": self.config.mode.value,
            "asset_class": "crypto",
        }
        self._log_trade(trade_record)
        return trade_record

    def _submit_equity_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str = "market",
        time_in_force: str = "day",
        limit_price: Optional[float] = None,
    ) -> dict:
        """Submit an equity order via alpaca-py SDK."""
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest

        order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
        tif = TimeInForce.DAY if time_in_force.lower() == "day" else TimeInForce.GTC

        if order_type.lower() == "limit" and limit_price:
            order_request = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=tif,
                limit_price=limit_price,
            )
        else:
            order_request = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=tif,
            )

        order = self._trading_client.submit_order(order_request)

        trade_record = {
            "order_id": str(order.id),
            "symbol": symbol,
            "qty": qty,
            "side": side,
            "order_type": order_type,
            "status": order.status.value if hasattr(order.status, 'value') else str(order.status),
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "mode": self.config.mode.value,
            "asset_class": "equity",
        }
        self._log_trade(trade_record)
        return trade_record

    def submit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str = "market",
        time_in_force: str = "day",
        limit_price: Optional[float] = None,
    ) -> dict:
        if self._is_crypto(symbol):
            return self._submit_crypto_order(symbol, qty, side, order_type, time_in_force, limit_price)
        return self._submit_equity_order(symbol, qty, side, order_type, time_in_force, limit_price)

    def get_order_status(self, order_id: str) -> dict:
        try:
            from alpaca.trading.requests import GetOrderRequest

            request = GetOrderRequest(order_id=order_id)
            order = self._trading_client.get_order_by_id(request)
            return {
                "order_id": str(order.id),
                "status": order.status.value if hasattr(order.status, 'value') else str(order.status),
                "filled_qty": int(float(order.filled_qty)) if order.filled_qty else 0,
                "filled_avg_price": float(order.filled_avg_price) if order.filled_avg_price else None,
                "filled_at": order.filled_at.isoformat() if order.filled_at else None,
            }
        except Exception as e:
            return {"order_id": order_id, "error": str(e)}

    def list_orders(self, status: str = "all", limit: int = 100) -> list[dict]:
        try:
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus

            if status.lower() == "all":
                query_status = None
            elif status.lower() == "open":
                query_status = QueryOrderStatus.OPEN
            elif status.lower() == "closed":
                query_status = QueryOrderStatus.CLOSED
            else:
                query_status = None

            request = GetOrdersRequest(status=query_status, limit=limit)
            orders = self._trading_client.get_orders(request)
        except Exception:
            return []

        payload = []
        for order in orders:
            payload.append(
                {
                    "order_id": str(order.id),
                    "symbol": order.symbol,
                    "qty": int(float(order.qty)) if order.qty else 0,
                    "filled_qty": int(float(order.filled_qty)) if order.filled_qty else 0,
                    "side": order.side.value if hasattr(order.side, 'value') else str(order.side),
                    "order_type": order.order_type.value if hasattr(order.order_type, 'value') else str(order.order_type),
                    "status": order.status.value if hasattr(order.status, 'value') else str(order.status),
                    "submitted_at": order.submitted_at.isoformat() if order.submitted_at else None,
                    "filled_at": order.filled_at.isoformat() if order.filled_at else None,
                    "filled_avg_price": float(order.filled_avg_price) if order.filled_avg_price else None,
                    "mode": self.config.mode.value,
                }
            )
        return payload

    def cancel_order(self, order_id: str) -> bool:
        try:
            from alpaca.trading.requests import CancelOrderRequest

            request = CancelOrderRequest(order_id=order_id)
            self._trading_client.cancel_order_by_id(request)
            return True
        except Exception:
            logger.exception("Failed to cancel Alpaca order %s", order_id)
            return False

    def check_adv_limit(
        self,
        symbol: str,
        qty: int,
        price: float,
        fraction: float = 0.05,
        lookback_days: int = 20,
    ) -> bool:
        try:
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame
            from datetime import timedelta

            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Minute_15,
                limit=min(320, lookback_days * 16),
            )
            bars_response = self._data_client.get_stock_bars(request)

            if symbol not in bars_response:
                return True

            bars = bars_response[symbol]
            if not bars:
                return True

            total_volume = sum(b.volume for b in bars)
            avg_volume = total_volume / len(bars) if bars else 0
            adv = avg_volume * price * lookback_days

            return (qty * price) <= (fraction * adv)
        except Exception:
            logger.exception(
                "Failed to check ADV limit for %s qty=%s price=%s",
                symbol,
                qty,
                price,
            )
            return True

    def check_position_limits(
        self,
        symbol: str,
        proposed_qty: int,
        max_position_value: Optional[float] = None,
    ) -> tuple[bool, str]:
        if max_position_value is None:
            max_position_value = float(os.getenv("MAX_POSITION_SIZE", "50000"))

        position = self.get_position(symbol)
        if position is None:
            current_value = 0.0
            estimated_price = self.get_latest_price(symbol)
            if estimated_price is None:
                return False, f"Cannot verify position size for {symbol}: price unavailable"
            new_value = proposed_qty * estimated_price
        else:
            current_price = position.get("current_price", position.get("avg_price", 0))
            current_value = position["qty"] * current_price
            new_value = current_value + proposed_qty * current_price

        if new_value > max_position_value:
            return False, f"Would exceed max position size of ${max_position_value}"
        return True, "OK"

    def _log_trade(self, trade_record: dict) -> None:
        date_str = datetime.now().strftime("%Y-%m-%d")
        log_file = self._trade_log_dir / f"{date_str}.json"

        with _TRADE_LOG_LOCK:
            existing = []
            if log_file.exists():
                try:
                    existing = json.loads(log_file.read_text())
                except Exception:
                    existing = []

            existing.append(trade_record)
            log_file.write_text(json.dumps(existing, indent=2))

    def get_trade_log(self, date: Optional[str] = None) -> list[dict]:
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        log_file = self._trade_log_dir / f"{date}.json"
        if not log_file.exists():
            return []
        try:
            return json.loads(log_file.read_text())
        except Exception:
            return []

    def liquidate_position(self, symbol: str) -> dict:
        position = self.get_position(symbol)
        if position is None:
            return {"symbol": symbol, "status": "no_position"}
        return self.submit_order(symbol, position["qty"], "sell")

    def liquidate_all(self) -> list[dict]:
        results = []
        positions = self.get_all_positions()
        for _, pos in positions.iterrows():
            results.append(self.submit_order(pos["symbol"], pos["qty"], "sell"))
        return results


def create_alpaca_node(mode: Optional[str] = None) -> AlpacaLiveNode:
    if mode:
        os.environ["TRADING_MODE"] = mode
    config = AlpacaConfig.from_env()
    return AlpacaLiveNode(config)
