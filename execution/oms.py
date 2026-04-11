"""Order Management System (OMS) for MLCouncil.

Provides complete order lifecycle management:
- Order creation, submission, tracking
- Fill aggregation and attribution
- Partial fill handling
- Order modification and cancellation
- Execution quality analysis

Usage:
    from execution.oms import OrderManager, Order, OrderStatus

    oms = OrderManager()
    order = oms.create_order("AAPL", 100, "buy", order_type="market")
    oms.submit(order, broker)
    oms.track_fills(order)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

import pandas as pd

_ROOT = Path(__file__).parents[1]
ORDERS_DIR = _ROOT / "data" / "orders"
OMS_DIR = _ROOT / "data" / "oms"
OMS_DIR.mkdir(parents=True, exist_ok=True)


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TWAP = "twap"
    VWAP = "vwap"


class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class TimeInForce(Enum):
    DAY = "day"
    GTC = "gtc"
    IOC = "ioc"
    FOK = "fok"


@dataclass
class Fill:
    fill_id: str
    order_id: str
    symbol: str
    quantity: int
    price: float
    commission: float
    timestamp: datetime
    venue: str = "ALPACA"

    @property
    def notional(self) -> float:
        return self.quantity * self.price

    @property
    def net_notional(self) -> float:
        return self.notional + self.commission


@dataclass
class Order:
    order_id: str
    symbol: str
    quantity: int
    side: OrderSide
    order_type: OrderType = OrderType.MARKET
    time_in_force: TimeInForce = TimeInForce.DAY
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    avg_fill_price: Optional[float] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    child_orders: list[str] = field(default_factory=list)
    parent_order_id: Optional[str] = None
    tags: dict = field(default_factory=dict)

    @property
    def remaining_quantity(self) -> int:
        return self.quantity - self.filled_quantity

    @property
    def is_complete(self) -> bool:
        return self.status in {OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED}

    @property
    def fill_rate(self) -> float:
        if self.quantity == 0:
            return 1.0
        return self.filled_quantity / self.quantity

    def to_dict(self) -> dict:
        d = asdict(self)
        d["side"] = self.side.value
        d["order_type"] = self.order_type.value
        d["time_in_force"] = self.time_in_force.value
        d["status"] = self.status.value
        d["created_at"] = self.created_at.isoformat()
        d["updated_at"] = self.updated_at.isoformat()
        if self.submitted_at:
            d["submitted_at"] = self.submitted_at.isoformat()
        if self.filled_at:
            d["filled_at"] = self.filled_at.isoformat()
        return d


class OrderManager:
    def __init__(self, orders_dir: Path = ORDERS_DIR):
        self.orders_dir = orders_dir
        self.orders: dict[str, Order] = {}
        self.fills: dict[str, list[Fill]] = {}
        self._load_pending_orders()

    def _load_pending_orders(self):
        pending_file = OMS_DIR / "pending_orders.json"
        if pending_file.exists():
            try:
                data = json.loads(pending_file.read_text())
                for d in data:
                    d["side"] = OrderSide(d["side"])
                    d["order_type"] = OrderType(d["order_type"])
                    d["time_in_force"] = TimeInForce(d["time_in_force"])
                    d["status"] = OrderStatus(d["status"])
                    d["created_at"] = datetime.fromisoformat(d["created_at"])
                    d["updated_at"] = datetime.fromisoformat(d["updated_at"])
                    if d["submitted_at"]:
                        d["submitted_at"] = datetime.fromisoformat(d["submitted_at"])
                    if d["filled_at"]:
                        d["filled_at"] = datetime.fromisoformat(d["filled_at"])
                    order = Order(**d)
                    self.orders[order.order_id] = order
            except Exception:
                pass

    def _save_pending_orders(self):
        pending_file = OMS_DIR / "pending_orders.json"
        pending_file.write_text(json.dumps([o.to_dict() for o in self.orders.values() if not o.is_complete], indent=2))

    def _generate_order_id(self, symbol: str) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
        return f"{symbol}_{timestamp}"

    def _generate_fill_id(self, order_id: str, fill_seq: int) -> str:
        return f"{order_id}_F{fill_seq:03d}"

    def create_order(
        self,
        symbol: str,
        quantity: int,
        side: str,
        order_type: str = "market",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = "day",
        tags: Optional[dict] = None,
    ) -> Order:
        order_id = self._generate_order_id(symbol)
        order = Order(
            order_id=order_id,
            symbol=symbol,
            quantity=abs(quantity),
            side=OrderSide(side.lower()),
            order_type=OrderType(order_type.lower()),
            time_in_force=TimeInForce(time_in_force.lower()),
            limit_price=limit_price,
            stop_price=stop_price,
            tags=tags or {},
        )
        self.orders[order_id] = order
        self.fills[order_id] = []
        self._save_pending_orders()
        return order

    def submit(self, order: Order, broker: "BrokerAdapter") -> bool:
        try:
            order.status = OrderStatus.SUBMITTED
            order.submitted_at = datetime.now(timezone.utc)
            order.updated_at = datetime.now(timezone.utc)

            result = broker.submit_order(
                symbol=order.symbol,
                quantity=order.quantity,
                side=order.side.value,
                order_type=order.order_type.value,
                time_in_force=order.time_in_force.value,
                limit_price=order.limit_price,
                stop_price=order.stop_price,
            )

            if result and result.get("order_id"):
                order.tags["broker_order_id"] = result["order_id"]
                self._save_pending_orders()
                return True

            order.status = OrderStatus.REJECTED
            order.tags["rejection_reason"] = result.get("error", "Unknown error")
            self._save_pending_orders()
            return False

        except Exception as e:
            order.status = OrderStatus.REJECTED
            order.tags["rejection_reason"] = str(e)
            self._save_pending_orders()
            return False

    def update_from_broker(self, order: Order, broker_status: dict) -> bool:
        try:
            status_map = {
                "new": OrderStatus.SUBMITTED,
                "accepted": OrderStatus.SUBMITTED,
                "pending_new": OrderStatus.SUBMITTED,
                "partially_filled": OrderStatus.PARTIAL,
                "filled": OrderStatus.FILLED,
                "done_for_day": OrderStatus.FILLED,
                "cancelled": OrderStatus.CANCELLED,
                "pending_cancel": OrderStatus.CANCELLED,
                "pending_replace": OrderStatus.SUBMITTED,
                "rejected": OrderStatus.REJECTED,
                "expired": OrderStatus.EXPIRED,
            }

            new_status_str = broker_status.get("status", "").lower()
            new_status = status_map.get(new_status_str, OrderStatus.SUBMITTED)

            if new_status == OrderStatus.PARTIAL:
                order.filled_quantity = int(broker_status.get("filled_qty", 0))
                order.avg_fill_price = float(broker_status.get("filled_avg_price", 0))
            elif new_status == OrderStatus.FILLED:
                order.filled_quantity = int(broker_status.get("filled_qty", order.quantity))
                order.avg_fill_price = float(broker_status.get("filled_avg_price", 0))
                order.filled_at = datetime.now(timezone.utc)

            order.status = new_status
            order.updated_at = datetime.now(timezone.utc)
            self._save_pending_orders()
            return True

        except Exception:
            return False

    def add_fill(self, order: Order, fill: Fill) -> None:
        self.fills[order.order_id].append(fill)
        order.filled_quantity += fill.quantity

        if order.avg_fill_price is None:
            order.avg_fill_price = fill.price
        else:
            total_value = (order.avg_fill_price * (order.filled_quantity - fill.quantity)) + (fill.price * fill.quantity)
            order.avg_fill_price = total_value / order.filled_quantity

        if order.filled_quantity >= order.quantity:
            order.status = OrderStatus.FILLED
            order.filled_at = datetime.now(timezone.utc)

        order.updated_at = datetime.now(timezone.utc)
        self._save_pending_orders()
        self._save_fill(fill)

    def _save_fill(self, fill: Fill) -> None:
        fills_file = OMS_DIR / f"fills_{fill.order_id}.json"
        fills_file.write_text(json.dumps([asdict(fill)], indent=2))

    def cancel(self, order: Order, broker: "BrokerAdapter") -> bool:
        if order.is_complete:
            return False

        try:
            broker_order_id = order.tags.get("broker_order_id")
            if broker_order_id:
                success = broker.cancel_order(broker_order_id)
                if success:
                    order.status = OrderStatus.CANCELLED
                    order.updated_at = datetime.now(timezone.utc)
                    self._save_pending_orders()
                    return True
            return False
        except Exception:
            return False

    def slice_order_twap(
        self,
        parent_order: Order,
        n_slices: int = 8,
    ) -> list[Order]:
        slice_qty = parent_order.quantity // n_slices
        child_orders = []

        for i in range(n_slices):
            child = self.create_order(
                symbol=parent_order.symbol,
                quantity=slice_qty + (1 if i < (parent_order.quantity % n_slices) else 0),
                side=parent_order.side.value,
                order_type="limit",
                limit_price=parent_order.limit_price,
                time_in_force=parent_order.time_in_force.value,
                tags={**parent_order.tags, "parent_order_id": parent_order.order_id, "slice": i},
            )
            child.parent_order_id = parent_order.order_id
            child_orders.append(child)
            parent_order.child_orders.append(child.order_id)

        return child_orders

    def get_order(self, order_id: str) -> Optional[Order]:
        return self.orders.get(order_id)

    def get_pending_orders(self) -> list[Order]:
        return [o for o in self.orders.values() if not o.is_complete]

    def get_order_history(
        self,
        symbol: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        records = []
        for order in self.orders.values():
            if symbol and order.symbol != symbol:
                continue
            if start_date and order.created_at < start_date:
                continue
            if end_date and order.created_at > end_date:
                continue
            record = order.to_dict()
            record["fills"] = len(self.fills.get(order.order_id, []))
            records.append(record)

        return pd.DataFrame(records)

    def compute_execution_quality(self, order_id: str, reference_price: float) -> dict:
        order = self.orders.get(order_id)
        if not order or not order.is_complete:
            return {}

        fills = self.fills.get(order_id, [])
        if not fills:
            return {}

        total_cost = sum(f.net_notional for f in fills)
        avg_price = order.avg_fill_price or 0

        if order.side == OrderSide.BUY:
            slippage = (avg_price - reference_price) / reference_price
            execution_cost = total_cost - (order.quantity * reference_price)
        else:
            slippage = (reference_price - avg_price) / reference_price
            execution_cost = (order.quantity * reference_price) - total_cost

        return {
            "order_id": order_id,
            "symbol": order.symbol,
            "side": order.side.value,
            "quantity": order.quantity,
            "filled_quantity": order.filled_quantity,
            "avg_fill_price": avg_price,
            "reference_price": reference_price,
            "slippage_bps": slippage * 10000,
            "execution_cost": execution_cost,
            "execution_cost_bps": (execution_cost / (order.quantity * reference_price)) * 10000 if reference_price > 0 else 0,
            "fill_rate": order.fill_rate,
            "n_fills": len(fills),
        }


class BrokerAdapter:
    def submit_order(self, **kwargs) -> dict:
        raise NotImplementedError

    def cancel_order(self, broker_order_id: str) -> bool:
        raise NotImplementedError

    def get_order_status(self, broker_order_id: str) -> dict:
        raise NotImplementedError


class AlpacaBrokerAdapter(BrokerAdapter):
    def __init__(self, api_key: str, api_secret: str, paper: bool = True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.paper = paper
        self._trading_client = None  # Lazily initialised; cached for connection-pool reuse.

    def _get_client(self):
        if self._trading_client is None:
            from alpaca.trading.client import TradingClient
            self._trading_client = TradingClient(self.api_key, self.api_secret, paper=self.paper)
        return self._trading_client

    def submit_order(self, **kwargs) -> dict:
        try:
            from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
            from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest

            client = self._get_client()

            symbol = kwargs.get("symbol")
            quantity = kwargs.get("quantity") or kwargs.get("qty")
            side = OrderSide.BUY if str(kwargs.get("side", "buy")).lower() == "buy" else OrderSide.SELL
            order_type = str(kwargs.get("order_type", "market")).lower()
            time_in_force = TimeInForce.DAY if str(kwargs.get("time_in_force", "day")).lower() == "day" else TimeInForce.GTC

            if order_type == "limit" and kwargs.get("limit_price"):
                order_request = LimitOrderRequest(
                    symbol=symbol,
                    qty=quantity,
                    side=side,
                    time_in_force=time_in_force,
                    limit_price=kwargs.get("limit_price"),
                )
            else:
                order_request = MarketOrderRequest(
                    symbol=symbol,
                    qty=quantity,
                    side=side,
                    time_in_force=time_in_force,
                )

            order = client.submit_order(order_request)
            status_value = order.status.value if hasattr(order.status, 'value') else str(order.status)
            return {"order_id": str(order.id), "status": status_value}
        except Exception as e:
            return {"error": str(e)}

    def cancel_order(self, broker_order_id: str) -> bool:
        try:
            from alpaca.trading.requests import CancelOrderRequest
            client = self._get_client()
            request = CancelOrderRequest(order_id=broker_order_id)
            client.cancel_order_by_id(request)
            return True
        except Exception:
            return False

    def get_order_status(self, broker_order_id: str) -> dict:
        try:
            from alpaca.trading.requests import GetOrderRequest
            client = self._get_client()
            request = GetOrderRequest(order_id=broker_order_id)
            order = client.get_order_by_id(request)
            status_value = order.status.value if hasattr(order.status, 'value') else str(order.status)
            return {
                "order_id": str(order.id),
                "status": status_value.lower(),
                "filled_qty": int(float(order.filled_qty)) if order.filled_qty else 0,
                "filled_avg_price": float(order.filled_avg_price) if order.filled_avg_price else None,
            }
        except Exception as e:
            return {"error": str(e)}
