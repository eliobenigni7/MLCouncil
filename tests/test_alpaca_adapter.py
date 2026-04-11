from __future__ import annotations

from unittest.mock import MagicMock, patch


def _make_node():
    from execution.alpaca_adapter import AlpacaConfig, AlpacaLiveNode, TradingMode

    node = AlpacaLiveNode.__new__(AlpacaLiveNode)
    node.config = AlpacaConfig(
        paper_key="paper-key",
        paper_secret="paper-secret",
        mode=TradingMode.PAPER,
    )
    node._trading_client = MagicMock()
    node._data_client = MagicMock()
    return node


def test_cancel_order_logs_exception_and_returns_false():
    node = _make_node()
    node._trading_client.cancel_order_by_id.side_effect = RuntimeError("cancel failed")

    with patch("execution.alpaca_adapter.logger.exception") as log_exception:
        result = node.cancel_order("order-123")

    assert result is False
    log_exception.assert_called_once()


def test_check_adv_limit_logs_exception_and_returns_true():
    node = _make_node()
    node._data_client.get_stock_bars.side_effect = RuntimeError("bars unavailable")

    with patch("execution.alpaca_adapter.logger.exception") as log_exception:
        result = node.check_adv_limit("AAPL", qty=10, price=150.0)

    assert result is True
    log_exception.assert_called_once()
