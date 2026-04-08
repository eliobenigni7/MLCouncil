"""Trading service — executes orders to Alpaca Paper with safety guards."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import pandas as pd

if TYPE_CHECKING:
    from execution.alpaca_adapter import AlpacaLiveNode, AlpacaConfig, TradingMode

_ROOT = Path(__file__).parents[2]
TRADE_LOG_DIR = _ROOT / "data" / "paper_trades"
ORDERS_DIR = _ROOT / "data" / "orders"
_TRADE_LOG_LOCK = threading.Lock()


class TradingService:
    def __init__(self):
        from runtime_env import load_runtime_env
        load_runtime_env()
        self._node: Optional["AlpacaLiveNode"] = None
        self._config: Optional["AlpacaConfig"] = None

    def _get_config(self) -> "AlpacaConfig":
        if self._config is None:
            from execution.alpaca_adapter import AlpacaConfig
            self._config = AlpacaConfig.from_env()
        return self._config

    @property
    def node(self) -> "AlpacaLiveNode":
        if self._node is None:
            from execution.alpaca_adapter import AlpacaLiveNode
            config = self._get_config()
            self._node = AlpacaLiveNode(config)
        return self._node

    @property
    def is_paper(self) -> bool:
        config = self._get_config()
        return config.mode.value == "paper"

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
        max_position = float(os.getenv("MLCOUNCIL_MAX_POSITION_SIZE", "0.10"))
        portfolio_value = float(account.get("portfolio_value", 0))
        if portfolio_value <= 0:
            return False, "No buying power"

        symbol = order.get("ticker")
        direction = order.get("direction", "buy").lower()
        quantity = order.get("quantity", 0)

        if quantity <= 0:
            return False, f"Invalid quantity for {symbol}"

        if direction == "buy":
            # Use target_weight from the optimizer output; fall back to
            # quantity/portfolio_value for legacy order formats that include price.
            target_weight = order.get("target_weight")
            if target_weight is not None:
                weight = float(target_weight)
            else:
                price = order.get("price", 0)
                weight = (quantity * price) / portfolio_value if price > 0 else 0.0
            if weight > max_position:
                return False, f"{symbol} exceeds max position {max_position:.0%}"

        return True, "OK"

    def execute_orders(self, date: str) -> dict:
        """Execute pending orders for date to Alpaca Paper."""
        if not self.is_paper:
            return {"error": "Live trading blocked — paper mode only"}

        try:
            account = self.node.get_account_info()
            positions_df = self.node.get_all_positions()
            orders = self.get_pending_orders(date)
            lineage = self._extract_lineage(orders)

            if not orders:
                return {"error": f"No orders found for {date}"}

            max_daily = int(os.getenv("MLCOUNCIL_MAX_DAILY_ORDERS", "20"))
            if len(orders) > max_daily:
                return {"error": f"Order count {len(orders)} exceeds max {max_daily}"}

            target_tickers = {o["ticker"] for o in orders}

            liquidate_results = []
            if not positions_df.empty:
                for _, pos in positions_df.iterrows():
                    if pos["symbol"] not in target_tickers:
                        result = self.node.submit_order(pos["symbol"], int(pos["qty"]), "sell")
                        liquidate_results.append(result)

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
                "lineage": lineage,
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
        with _TRADE_LOG_LOCK:
            existing = []
            if path.exists():
                try:
                    existing = json.loads(path.read_text())
                except Exception:
                    existing = []
            existing.append(data)
            path.write_text(json.dumps(existing, indent=2))

    def _extract_lineage(self, orders: list[dict]) -> dict[str, str]:
        if not orders:
            return {}

        keys = (
            "pipeline_run_id",
            "data_version",
            "feature_version",
            "model_version",
        )
        first = orders[0]
        return {
            key: str(first[key])
            for key in keys
            if key in first and first[key] is not None
        }


service = TradingService()
