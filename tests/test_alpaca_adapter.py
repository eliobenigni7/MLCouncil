from __future__ import annotations

import importlib
import sys
import types


def _load_adapter(monkeypatch, tmp_path):
    class FakeREST:
        def __init__(self, key, secret, base_url):
            self.key = key
            self.secret = secret
            self.base_url = base_url
            self.orders = []

        def get_account(self):
            return types.SimpleNamespace(
                buying_power="100000",
                cash="50000",
                portfolio_value="125000",
                status="ACTIVE",
            )

        def list_positions(self):
            return []

        def submit_order(self, **kwargs):
            self.orders.append(kwargs)
            return types.SimpleNamespace(id="ord-123", status="accepted")

        def get_latest_trade(self, symbol):
            return types.SimpleNamespace(price=123.45)

        def get_asset(self, symbol):
            return types.SimpleNamespace(tradable=True, fractionable=True)

    fake_module = types.SimpleNamespace(REST=FakeREST)
    monkeypatch.setitem(sys.modules, "alpaca_trade_api", fake_module)
    monkeypatch.setenv("ALPACA_PAPER_KEY", "paper-key")
    monkeypatch.setenv("ALPACA_PAPER_SECRET", "paper-secret")
    monkeypatch.setenv("TRADING_MODE", "paper")
    monkeypatch.setenv("MLCOUNCIL_RUNTIME_ENV_PATH", str(tmp_path / "runtime.env"))

    import execution.alpaca_adapter as adapter

    importlib.reload(adapter)
    monkeypatch.setattr(adapter, "_ROOT", tmp_path)
    return adapter


def test_alpaca_config_from_env_uses_paper_keys(monkeypatch, tmp_path):
    adapter = _load_adapter(monkeypatch, tmp_path)

    config = adapter.AlpacaConfig.from_env()

    assert config.paper_key == "paper-key"
    assert config.paper_secret == "paper-secret"
    assert config.mode.value == "paper"


def test_alpaca_live_node_uses_paper_endpoint(monkeypatch, tmp_path):
    adapter = _load_adapter(monkeypatch, tmp_path)

    node = adapter.AlpacaLiveNode(adapter.AlpacaConfig.from_env())

    assert node.is_paper is True
    assert node._api.base_url == "https://paper-api.alpaca.markets"


def test_get_latest_price_returns_float(monkeypatch, tmp_path):
    adapter = _load_adapter(monkeypatch, tmp_path)
    node = adapter.AlpacaLiveNode(adapter.AlpacaConfig.from_env())

    assert node.get_latest_price("AAPL") == 123.45


def test_submit_order_logs_trade(monkeypatch, tmp_path):
    adapter = _load_adapter(monkeypatch, tmp_path)
    node = adapter.AlpacaLiveNode(adapter.AlpacaConfig.from_env())

    result = node.submit_order("AAPL", 5, "buy")
    log_entries = node.get_trade_log()

    assert result["status"] == "accepted"
    assert log_entries
    assert log_entries[0]["symbol"] == "AAPL"
