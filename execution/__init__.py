"""Execution layer for MLCouncil trading system.

Modules:
    alpaca_adapter: Alpaca broker adapter for paper/live trading.
    slicer: TWAP/VWAP order slicing for large orders.
    oms: Order Management System with full lifecycle tracking.

Usage:
    from execution import AlpacaLiveNode, AlpacaConfig, OrderManager

    config = AlpacaConfig.from_env()
    node = AlpacaLiveNode(config)
    oms = OrderManager()
"""

from execution.alpaca_adapter import (
    AlpacaConfig,
    AlpacaLiveNode,
    TradingMode,
    create_alpaca_node,
)

from execution.slicer import OrderSlicer, ChildOrder, estimate_adv

from execution.oms import (
    OrderManager,
    Order,
    OrderStatus,
    OrderSide,
    OrderType,
    TimeInForce,
    Fill,
    BrokerAdapter,
    AlpacaBrokerAdapter,
)

__all__ = [
    "AlpacaConfig",
    "AlpacaLiveNode",
    "TradingMode",
    "create_alpaca_node",
    "OrderSlicer",
    "ChildOrder",
    "estimate_adv",
    "OrderManager",
    "Order",
    "OrderStatus",
    "OrderSide",
    "OrderType",
    "TimeInForce",
    "Fill",
    "BrokerAdapter",
    "AlpacaBrokerAdapter",
]
