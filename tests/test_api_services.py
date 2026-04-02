import pytest
from pathlib import Path
from unittest.mock import patch


class TestConfigService:
    def test_get_universe_returns_dict(self):
        from api.services.config_service import get_universe
        result = get_universe()
        assert isinstance(result, dict)
        assert "universe" in result

    def test_get_models_returns_dict(self):
        from api.services.config_service import get_models
        result = get_models()
        assert isinstance(result, dict)
        assert "lgbm" in result


class TestMonitoringService:
    def test_get_current_alerts_no_file(self):
        from api.services.monitoring_service import get_current_alerts
        with patch("api.services.monitoring_service.CURRENT_ALERTS_PATH", Path("/nonexistent.json")):
            with patch("council.alerts.load_current_alerts", side_effect=FileNotFoundError):
                result = get_current_alerts()
        assert result == []

    def test_get_alert_history_no_dir(self):
        from api.services.monitoring_service import get_alert_history
        with patch("api.services.monitoring_service.ALERTS_DIR", Path("/nonexistent")):
            result = get_alert_history()
        assert result == []


class TestPortfolioService:
    def test_get_order_dates_no_dir(self):
        from api.services.portfolio_service import get_order_dates
        with patch("api.services.portfolio_service.ORDERS_DIR", Path("/nonexistent")):
            result = get_order_dates()
        assert result == []
