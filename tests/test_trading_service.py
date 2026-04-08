"""Tests for trading service."""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import tempfile


class TestTradingService:
    def test_validate_order_valid(self):
        from api.services.trading_service import TradingService
        
        svc = TradingService.__new__(TradingService)
        svc._node = None
        svc._config = None
        
        order = {"ticker": "AAPL", "direction": "buy", "quantity": 10, "price": 150.0}
        account = {"portfolio_value": 100000.0}
        positions = MagicMock()
        
        valid, msg = svc._validate_order(order, account, positions)
        assert valid is True

    def test_validate_order_exceeds_position_limit(self):
        from api.services.trading_service import TradingService
        
        svc = TradingService.__new__(TradingService)
        svc._node = None
        svc._config = None
        
        order = {"ticker": "AAPL", "direction": "buy", "quantity": 1000, "price": 150.0}
        account = {"portfolio_value": 100000.0}
        positions = MagicMock()
        
        valid, msg = svc._validate_order(order, account, positions)
        assert valid is False
        assert "exceeds max position" in msg

    def test_validate_order_no_buying_power(self):
        from api.services.trading_service import TradingService
        
        svc = TradingService.__new__(TradingService)
        svc._node = None
        svc._config = None
        
        order = {"ticker": "AAPL", "direction": "buy", "quantity": 10, "price": 150.0}
        account = {"portfolio_value": 0.0}
        positions = MagicMock()
        
        valid, msg = svc._validate_order(order, account, positions)
        assert valid is False
        assert "No buying power" in msg

    def test_validate_order_invalid_quantity(self):
        from api.services.trading_service import TradingService
        
        svc = TradingService.__new__(TradingService)
        svc._node = None
        svc._config = None
        
        order = {"ticker": "AAPL", "direction": "buy", "quantity": 0, "price": 150.0}
        account = {"portfolio_value": 100000.0}
        positions = MagicMock()
        
        valid, msg = svc._validate_order(order, account, positions)
        assert valid is False
        assert "Invalid quantity" in msg

    def test_get_pending_orders_empty(self):
        from api.services.trading_service import TradingService
        
        with tempfile.TemporaryDirectory() as tmpdir:
            svc = TradingService.__new__(TradingService)
            svc._node = None
            svc._config = None
            
            original_dir = Path("data/orders")
            import api.services.trading_service as ts
            ts.ORDERS_DIR = Path(tmpdir)
            
            result = svc.get_pending_orders("2024-01-01")
            assert result == []
            
            ts.ORDERS_DIR = original_dir

    def test_get_latest_order_date_no_files(self):
        from api.services.trading_service import TradingService
        
        with tempfile.TemporaryDirectory() as tmpdir:
            svc = TradingService.__new__(TradingService)
            svc._node = None
            svc._config = None
            
            import api.services.trading_service as ts
            ts.ORDERS_DIR = Path(tmpdir)
            
            result = svc.get_latest_order_date()
            assert result is None
            
            ts.ORDERS_DIR = Path("data/orders")

    def test_get_status_connected(self):
        from api.services.trading_service import TradingService
        
        svc = TradingService.__new__(TradingService)
        
        mock_node = MagicMock()
        mock_node.get_account_info.return_value = {"buying_power": 10000, "portfolio_value": 50000}
        mock_node.get_all_positions.return_value = MagicMock(empty=True)
        svc._node = mock_node
        
        mock_config = MagicMock()
        mock_config.mode.value = "paper"
        svc._config = mock_config
        
        status = svc.get_status()
        assert status["connected"] is True
        assert status["paper"] is True
        assert "account" in status

    def test_get_status_not_paper(self):
        from api.services.trading_service import TradingService
        
        svc = TradingService.__new__(TradingService)
        svc._node = None
        
        mock_config = MagicMock()
        mock_config.mode.value = "live"
        svc._config = mock_config
        
        status = svc.get_status()
        assert "error" in status
        assert "paper mode only" in status["error"]

    def test_execute_orders_returns_lineage_from_order_file(self):
        from api.services.trading_service import TradingService

        svc = TradingService.__new__(TradingService)

        mock_node = MagicMock()
        mock_node.get_account_info.return_value = {"portfolio_value": 100000.0}
        mock_node.get_all_positions.return_value = MagicMock(empty=True)
        mock_node.submit_order.return_value = {"status": "accepted"}
        svc._node = mock_node

        mock_config = MagicMock()
        mock_config.mode.value = "paper"
        svc._config = mock_config

        svc.get_pending_orders = MagicMock(
            return_value=[
                {
                    "ticker": "AAPL",
                    "direction": "buy",
                    "quantity": 10,
                    "target_weight": 0.10,
                    "pipeline_run_id": "run-004",
                    "data_version": "data-v4",
                    "feature_version": "feat-v4",
                    "model_version": "model-v4",
                }
            ]
        )

        result = svc.execute_orders("2024-01-15")

        assert result["lineage"] == {
            "pipeline_run_id": "run-004",
            "data_version": "data-v4",
            "feature_version": "feat-v4",
            "model_version": "model-v4",
        }
