from __future__ import annotations

from unittest.mock import MagicMock, patch

import json


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
    node._log_trade = MagicMock()
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


def test_submit_crypto_order_normalizes_unsupported_time_in_force_to_gtc():
    node = _make_node()

    class FakeResponse:
        ok = True

        def json(self):
            return {"id": "crypto-order-1", "status": "accepted"}

    captured: dict[str, object] = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        captured["timeout"] = timeout
        return FakeResponse()

    with patch("requests.post", side_effect=fake_post):
        result = node._submit_crypto_order("BTCUSD", 0.0785, "sell", time_in_force="day")

    assert result["status"] == "accepted"
    assert captured["json"]["symbol"] == "BTCUSD"
    assert captured["json"]["qty"] == "0.0785"
    assert captured["json"]["time_in_force"] == "gtc"


def test_list_orders_preserves_fractional_crypto_quantities():
    node = _make_node()

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return [
                {
                    "id": "crypto-open-1",
                    "symbol": "BTC/USD",
                    "qty": "0.07773475826327547",
                    "filled_qty": "0",
                    "side": "sell",
                    "order_type": "market",
                    "status": "pending_new",
                    "submitted_at": None,
                    "filled_at": None,
                    "filled_avg_price": None,
                }
            ]

    with patch("requests.get", return_value=FakeResponse()) as get_mock:
        payload = node.list_orders()

    assert get_mock.called
    assert payload[0]["symbol"] == "BTCUSD"
    assert payload[0]["qty"] == 0.07773475826327547
    assert payload[0]["filled_qty"] == 0.0


def test_get_all_positions_deduplicates_crypto_positions():
    import pandas as pd

    node = _make_node()

    equity_position = MagicMock()
    equity_position.symbol = "AAPL"
    equity_position.qty = "10"
    equity_position.avg_entry_price = "200"
    equity_position.current_price = "210"
    equity_position.market_value = "2100"
    equity_position.unrealized_pl = "100"
    equity_position.unrealized_plpc = "0.05"

    crypto_position = MagicMock()
    crypto_position.symbol = "BTCUSD"
    crypto_position.qty = "0.05"
    crypto_position.avg_entry_price = "70000"
    crypto_position.current_price = "71000"
    crypto_position.market_value = "3550"
    crypto_position.unrealized_pl = "50"
    crypto_position.unrealized_plpc = "0.014"

    node._trading_client.get_all_positions.return_value = [equity_position, crypto_position]
    node.get_crypto_positions = MagicMock(
        return_value=pd.DataFrame(
            [
                {
                    "symbol": "BTCUSD",
                    "qty": 0.05,
                    "avg_price": 70000.0,
                    "current_price": 71000.0,
                    "current_value": 3550.0,
                    "unrealized_pnl": 50.0,
                    "unrealized_pnl_pct": 0.014,
                    "asset_class": "crypto",
                }
            ]
        )
    )

    positions = node.get_all_positions()

    assert positions["symbol"].tolist() == ["AAPL", "BTCUSD"]
    assert positions["asset_class"].tolist() == ["equity", "crypto"]


def test_get_crypto_positions_preserves_fractional_quantity_precision():
    node = _make_node()

    class FakeResponse:
        ok = True
        status_code = 200

        def json(self):
            return [
                {
                    "symbol": "BTCUSD",
                    "qty": "0.00260128",
                    "avg_entry_price": "73728.660702",
                    "current_price": "70729.5",
                    "market_value": "183.976978",
                    "unrealized_pl": "-7.801222",
                    "unrealized_plpc": "-0.04068",
                }
            ]

    with patch("requests.get", return_value=FakeResponse()):
        positions = node.get_crypto_positions()

    assert not positions.empty
    assert positions.iloc[0]["qty"] == 0.00260128
