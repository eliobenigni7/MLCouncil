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
            paper_key=os.getenv("ALPACA_PAPER_KEY", ""),
            paper_secret=os.getenv("ALPACA_PAPER_SECRET", ""),
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

        self._api = None
        self._trade_log_dir = _ROOT / "data" / "paper_trades"
        self._trade_log_dir.mkdir(parents=True, exist_ok=True)

        self._initialize_api()

    def _initialize_api(self) -> None:
        try:
            import alpaca_trade_api as ata
        except ImportError:
            raise ImportError(
                "alpaca_trade_api is required for live trading. "
                "Install with: pip install alpaca-trade-api"
            )

        if self.config.mode == TradingMode.PAPER:
            base_url = "https://paper-api.alpaca.markets"
            key = self.config.paper_key
            secret = self.config.paper_secret
        else:
            base_url = "https://api.alpaca.markets"
            key = self.config.live_key
            secret = self.config.live_secret

        self._api = ata.REST(key, secret, base_url)

    @property
    def is_paper(self) -> bool:
        return self.config.mode == TradingMode.PAPER

    def is_tradeable(self, symbol: str) -> bool:
        try:
            asset = self._api.get_asset(symbol)
            return asset.tradable and asset.fractionable
        except Exception:
            return False

    def get_position(self, symbol: str) -> Optional[dict]:
        try:
            pos = self._api.get_position(symbol)
            return {
                "symbol": symbol,
                "qty": int(pos.qty),
                "avg_price": float(pos.avg_entry_price),
                "current_value": float(pos.market_value),
            }
        except Exception:
            return None

    def get_all_positions(self) -> pd.DataFrame:
        try:
            positions = self._api.list_positions()
            if not positions:
                return pd.DataFrame()
            return pd.DataFrame([
                {
                    "symbol": p.symbol,
                    "qty": int(p.qty),
                    "avg_price": float(p.avg_entry_price),
                    "current_price": float(p.current_price),
                    "current_value": float(p.market_value),
                    "unrealized_pnl": float(p.unrealized_pl),
                    "unrealized_pnl_pct": float(p.unrealized_pl_pc),
                }
                for p in positions
            ])
        except Exception:
            return pd.DataFrame()

    def get_account_info(self) -> dict:
        account = self._api.get_account()
        return {
            "buying_power": float(account.buying_power),
            "cash": float(account.cash),
            "portfolio_value": float(account.portfolio_value),
            "status": account.status,
            "mode": self.config.mode.value,
        }

    def submit_order(
        self,
        symbol: str,
        qty: int,
        side: str,
        order_type: str = "market",
        time_in_force: str = "day",
        limit_price: Optional[float] = None,
    ) -> dict:
        order_params = {
            "symbol": symbol,
            "qty": qty,
            "side": side,
            "type": order_type,
            "time_in_force": time_in_force,
        }
        if limit_price:
            order_params["limit_price"] = str(limit_price)

        order = self._api.submit_order(**order_params)

        trade_record = {
            "order_id": order.id,
            "symbol": symbol,
            "qty": qty,
            "side": side,
            "order_type": order_type,
            "status": order.status,
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "mode": self.config.mode.value,
        }

        self._log_trade(trade_record)
        return trade_record

    def get_order_status(self, order_id: str) -> dict:
        try:
            order = self._api.get_order(order_id)
            return {
                "order_id": order.id,
                "status": order.status,
                "filled_qty": int(order.filled_qty) if order.filled_qty else 0,
                "filled_avg_price": float(order.filled_avg_price) if order.filled_avg_price else None,
                "filled_at": order.filled_at,
            }
        except Exception as e:
            return {"order_id": order_id, "error": str(e)}

    def cancel_order(self, order_id: str) -> bool:
        try:
            self._api.cancel_order(order_id)
            return True
        except Exception:
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
            bars = self._api.get_bars(
                symbol,
                "15Min",
                limit=min(320, lookback_days * 16),
            )
            if not bars:
                return True

            total_volume = sum(b.v for b in bars)
            avg_volume = total_volume / len(bars) if bars else 0
            adv = avg_volume * price * lookback_days

            return (qty * price) <= (fraction * adv)
        except Exception:
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
            # For a new position we need the current price to estimate order value.
            # Query the latest trade; if unavailable, reject conservatively.
            try:
                latest = self._api.get_latest_trade(symbol)
                estimated_price = float(latest.price)
            except Exception:
                return False, f"Cannot verify position size for {symbol}: price unavailable"
            new_value = proposed_qty * estimated_price
        else:
            current_price = position.get("current_price", position.get("avg_price", 0))
            current_value = position["qty"] * current_price
            estimated_price = current_value / max(position["qty"], 1)
            new_value = current_value + proposed_qty * estimated_price

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
