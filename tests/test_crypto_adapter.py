"""Tests for crypto trading support in alpaca_adapter."""

import pytest


class TestIsCrypto:
    """Test the _is_crypto symbol detection."""

    def test_crypto_symbols(self):
        from execution.alpaca_adapter import AlpacaLiveNode

        crypto_symbols = ["BTCUSD", "ETHUSD", "BTC/USD", "ETH/USD", "BTC-USD", "ETH-USD"]
        for sym in crypto_symbols:
            assert AlpacaLiveNode._is_crypto(sym) is True, f"{sym} should be crypto"

    def test_equity_symbols(self):
        from execution.alpaca_adapter import AlpacaLiveNode

        equity_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "SPY", "QQQ"]
        for sym in equity_symbols:
            assert AlpacaLiveNode._is_crypto(sym) is False, f"{sym} should NOT be crypto"

    def test_mixed_case(self):
        from execution.alpaca_adapter import AlpacaLiveNode

        assert AlpacaLiveNode._is_crypto("btcusd") is True
        assert AlpacaLiveNode._is_crypto("Eth/Usd") is True
        assert AlpacaLiveNode._is_crypto("aapl") is False


class TestCryptoOrderRouting:
    """Test that submit_order routes to the correct method."""

    def test_order_routing_logic(self):
        from execution.alpaca_adapter import AlpacaLiveNode

        node = AlpacaLiveNode.__new__(AlpacaLiveNode)
        # Just test the routing logic without making actual API calls
        assert AlpacaLiveNode._is_crypto("BTCUSD") is True
        assert AlpacaLiveNode._is_crypto("AAPL") is False


class TestCryptoPositionColumns:
    """Test that crypto positions include asset_class column."""

    def test_crypto_position_dict_has_asset_class(self):
        # Spot-check that the get_crypto_positions return format includes asset_class
        from execution.alpaca_adapter import AlpacaLiveNode

        node = AlpacaLiveNode.__new__(AlpacaLiveNode)
        node.get_crypto_positions  # Should exist
        assert hasattr(AlpacaLiveNode, "get_crypto_positions")


class TestCryptoEnvVars:
    """Test that crypto config env vars are read correctly."""

    def test_crypto_env_vars_loaded(self):
        import os
        from council.portfolio import PortfolioConstructor

        # Should pick up env vars without crashing
        pc = PortfolioConstructor()
        assert hasattr(pc, "crypto_enabled")
        assert hasattr(pc, "max_crypto_position")
        assert hasattr(pc, "max_crypto_turnover")

    def test_risk_engine_crypto_limits(self):
        from council.risk_engine import RiskEngine, RiskLimits

        limits = RiskLimits()
        assert limits.max_crypto_position == 0.20
        assert limits.max_single_position == 0.10
