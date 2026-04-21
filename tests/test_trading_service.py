"""Tests for trading service."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest


def _make_service(
    *,
    account: dict | None = None,
    positions: pd.DataFrame | None = None,
):
    from api.services.trading_service import TradingService

    os.environ["ALPACA_BASE_URL"] = "https://paper-api.alpaca.markets"

    svc = TradingService.__new__(TradingService)
    svc._node = MagicMock()
    svc._node.get_account_info.return_value = account or {
        "portfolio_value": 100000.0,
        "buying_power": 100000.0,
    }
    svc._node.get_all_positions.return_value = (
        positions if positions is not None else pd.DataFrame()
    )
    svc._node.get_latest_price.return_value = 200.0
    svc._node.submit_order.return_value = {"status": "accepted"}

    svc._config = MagicMock()
    svc._config.mode.value = "paper"
    return svc


class TestTradingService:
    def test_trading_service_uses_centralized_trading_settings(self, monkeypatch):
        from api.services import trading_service as ts
        from runtime_env import TradingSettings

        svc = _make_service()
        monkeypatch.delenv("MLCOUNCIL_AUTOMATION_PAUSED", raising=False)
        monkeypatch.setattr(
            ts,
            "get_trading_settings",
            lambda: TradingSettings(
                alpaca_base_url="https://paper-api.alpaca.markets",
                automation_paused=True,
                max_daily_orders=20,
                max_turnover=0.30,
                max_position_size=0.01,
                max_sector_exposure=0.25,
            ),
        )

        valid, msg = svc._validate_order(
            {"ticker": "AAPL", "direction": "buy", "quantity": 10, "price": 150.0},
            {"portfolio_value": 100000.0},
            pd.DataFrame(),
        )

        assert svc._is_automation_paused() is True
        assert valid is False
        assert "exceeds max position" in msg

    def test_validate_order_valid(self):
        svc = _make_service()

        order = {"ticker": "AAPL", "direction": "buy", "quantity": 10, "price": 150.0}
        account = {"portfolio_value": 100000.0}

        valid, msg = svc._validate_order(order, account, pd.DataFrame())
        assert valid is True
        assert msg == "OK"

    def test_validate_order_exceeds_position_limit(self):
        svc = _make_service()

        order = {"ticker": "AAPL", "direction": "buy", "quantity": 1000, "price": 150.0}
        account = {"portfolio_value": 100000.0}

        valid, msg = svc._validate_order(order, account, pd.DataFrame())
        assert valid is False
        assert "exceeds max position" in msg

    def test_validate_order_no_buying_power(self):
        svc = _make_service()

        order = {"ticker": "AAPL", "direction": "buy", "quantity": 10, "price": 150.0}
        account = {"portfolio_value": 0.0}

        valid, msg = svc._validate_order(order, account, pd.DataFrame())
        assert valid is False
        assert "No buying power" in msg

    def test_validate_order_invalid_quantity(self):
        svc = _make_service()

        order = {"ticker": "AAPL", "direction": "buy", "quantity": 0, "price": 150.0}
        account = {"portfolio_value": 100000.0}

        valid, msg = svc._validate_order(order, account, pd.DataFrame())
        assert valid is False
        assert "Invalid quantity" in msg

    def test_validate_order_accepts_fractional_crypto_quantity(self):
        svc = _make_service()

        order = {"ticker": "BTCUSD", "direction": "sell", "share_quantity": 0.075, "price": 70000.0}
        account = {"portfolio_value": 100000.0}

        valid, msg = svc._validate_order(order, account, pd.DataFrame())
        assert valid is True
        assert msg == "OK"

    def test_risk_adjust_intraday_orders_scales_to_position_cap(self, monkeypatch):
        svc = _make_service(
            account={"portfolio_value": 100000.0},
            positions=pd.DataFrame(
                [{"symbol": "BTCUSD", "current_value": 15000.0}]
            ),
        )
        monkeypatch.setenv("MLCOUNCIL_MAX_CRYPTO_POSITION_SIZE", "0.20")

        orders = [
            {
                "ticker": "BTCUSD",
                "direction": "buy",
                "quantity": 10000.0,
                "target_weight": 0.10,
            }
        ]

        adjusted, warnings = svc._risk_adjust_intraday_orders(orders)

        assert len(adjusted) == 1
        assert adjusted[0]["target_weight"] == pytest.approx(0.05)
        assert adjusted[0]["quantity"] == pytest.approx(5000.0)
        assert any("scaled" in item for item in warnings)

    def test_risk_adjust_intraday_orders_drops_buy_when_already_above_cap(self, monkeypatch):
        svc = _make_service(
            account={"portfolio_value": 100000.0},
            positions=pd.DataFrame(
                [{"symbol": "BTCUSD", "current_value": 30000.0}]
            ),
        )
        monkeypatch.setenv("MLCOUNCIL_MAX_CRYPTO_POSITION_SIZE", "0.20")

        orders = [
            {
                "ticker": "BTCUSD",
                "direction": "buy",
                "quantity": 2000.0,
                "target_weight": 0.02,
            }
        ]

        adjusted, warnings = svc._risk_adjust_intraday_orders(orders)

        assert adjusted == []
        assert any("dropped buy intent" in item for item in warnings)

    def test_get_pending_orders_empty(self):
        from api.services import trading_service as ts

        with tempfile.TemporaryDirectory() as tmpdir:
            svc = _make_service()
            original_dir = ts.ORDERS_DIR
            ts.ORDERS_DIR = Path(tmpdir)
            try:
                result = svc.get_pending_orders("2024-01-01")
            finally:
                ts.ORDERS_DIR = original_dir

        assert result == []

    def test_get_latest_order_date_no_files(self):
        from api.services import trading_service as ts

        with tempfile.TemporaryDirectory() as tmpdir:
            svc = _make_service()
            original_dir = ts.ORDERS_DIR
            ts.ORDERS_DIR = Path(tmpdir)
            try:
                result = svc.get_latest_order_date()
            finally:
                ts.ORDERS_DIR = original_dir

        assert result is None

    def test_get_trade_history_enriches_logs_with_live_broker_status(self):
        from api.services import trading_service as ts

        with tempfile.TemporaryDirectory() as tmpdir:
            svc = _make_service()
            svc._node.list_orders.return_value = [
                {
                    "order_id": "ord-123",
                    "symbol": "AAPL",
                    "status": "filled",
                    "filled_qty": 5,
                    "filled_avg_price": 199.5,
                    "submitted_at": "2026-04-08T17:47:31+00:00",
                    "filled_at": "2026-04-08T17:47:35+00:00",
                    "mode": "paper",
                }
            ]

            trade_dir = Path(tmpdir) / "paper_trades"
            trade_dir.mkdir(parents=True, exist_ok=True)
            (trade_dir / "2026-04-08.json").write_text(
                json.dumps(
                    [
                        {
                            "order_id": "ord-123",
                            "symbol": "AAPL",
                            "qty": 5,
                            "side": "buy",
                            "order_type": "market",
                            "status": "pending_new",
                            "submitted_at": "2026-04-08T17:47:31+00:00",
                            "mode": "paper",
                        }
                    ]
                )
            )

            original_dir = ts.TRADE_LOG_DIR
            ts.TRADE_LOG_DIR = trade_dir
            try:
                result = svc.get_trade_history(days=1)
            finally:
                ts.TRADE_LOG_DIR = original_dir

        assert len(result) == 1
        assert result[0]["order_id"] == "ord-123"
        assert result[0]["status"] == "filled"
        assert result[0]["filled_qty"] == 5

    def test_get_trade_history_returns_logs_when_broker_sync_fails(self):
        from api.services import trading_service as ts

        with tempfile.TemporaryDirectory() as tmpdir:
            svc = _make_service()
            svc._node.list_orders.side_effect = RuntimeError("broker offline")

            trade_dir = Path(tmpdir) / "paper_trades"
            trade_dir.mkdir(parents=True, exist_ok=True)
            (trade_dir / "2026-04-08.json").write_text(
                json.dumps(
                    [
                        {
                            "order_id": "ord-456",
                            "symbol": "MSFT",
                            "qty": 3,
                            "side": "buy",
                            "order_type": "market",
                            "status": "accepted",
                            "submitted_at": "2026-04-08T17:50:00+00:00",
                            "mode": "paper",
                        }
                    ]
                )
            )

            original_dir = ts.TRADE_LOG_DIR
            ts.TRADE_LOG_DIR = trade_dir
            try:
                result = svc.get_trade_history(days=1)
            finally:
                ts.TRADE_LOG_DIR = original_dir

        assert len(result) == 1
        assert result[0]["order_id"] == "ord-456"
        assert result[0]["status"] == "accepted"

    def test_get_trade_history_filters_out_non_trade_audit_payloads(self):
        from api.services import trading_service as ts

        with tempfile.TemporaryDirectory() as tmpdir:
            svc = _make_service()
            svc._node.list_orders.return_value = []

            trade_dir = Path(tmpdir) / "paper_trades"
            trade_dir.mkdir(parents=True, exist_ok=True)
            (trade_dir / "2026-04-13.json").write_text(
                json.dumps(
                    [
                        {
                            "account": {"portfolio_value": 100000.0},
                            "lineage": {"decision_id": "dec-1"},
                            "orders": [{"symbol": "BTCUSD"}],
                            "pretrade": {"blocked": False},
                            "reconciliation": {"date": "2026-04-13"},
                        },
                        {
                            "order_id": "ord-789",
                            "symbol": "BTCUSD",
                            "qty": 0.05,
                            "side": "sell",
                            "order_type": "market",
                            "status": "filled",
                            "submitted_at": "2026-04-13T09:18:22+00:00",
                            "mode": "paper",
                        },
                    ]
                )
            )

            original_dir = ts.TRADE_LOG_DIR
            ts.TRADE_LOG_DIR = trade_dir
            try:
                result = svc.get_trade_history(days=1)
            finally:
                ts.TRADE_LOG_DIR = original_dir

        assert len(result) == 1
        assert result[0]["order_id"] == "ord-789"
        assert result[0]["symbol"] == "BTCUSD"

    def test_get_status_connected_includes_runtime_flags(self, monkeypatch):
        monkeypatch.setenv("MLCOUNCIL_ENV_PROFILE", "paper")
        monkeypatch.setenv("MLCOUNCIL_AUTOMATION_PAUSED", "true")

        svc = _make_service()
        svc._node.get_all_positions.return_value = pd.DataFrame()

        status = svc.get_status()

        assert status["connected"] is True
        assert status["paper"] is True
        assert status["runtime_profile"] == "paper"
        assert status["paused"] is True
        assert "account" in status

    def test_get_status_not_paper(self):
        svc = _make_service()
        svc._config.mode.value = "live"

        status = svc.get_status()
        assert "error" in status
        assert "paper mode only" in status["error"]

    def test_build_pretrade_snapshot_blocks_when_paused(self, monkeypatch):
        monkeypatch.setenv("MLCOUNCIL_AUTOMATION_PAUSED", "true")

        svc = _make_service()
        svc.get_pending_orders = MagicMock(
            return_value=[
                {
                    "ticker": "AAPL",
                    "direction": "buy",
                    "quantity": 5000.0,
                    "target_weight": 0.05,
                    "pipeline_run_id": "run-001",
                    "data_version": "data-v1",
                    "feature_version": "feat-v1",
                    "model_version": "model-v1",
                }
            ]
        )

        snapshot = svc.build_pretrade_snapshot("2024-01-15")

        assert snapshot["paused"] is True
        assert snapshot["pretrade"]["blocked"] is True
        assert "paused" in snapshot["pretrade"]["reason"].lower()

    def test_build_pretrade_snapshot_blocks_on_high_risk_breach(self, monkeypatch):
        monkeypatch.setenv("MLCOUNCIL_AUTOMATION_PAUSED", "false")

        svc = _make_service()
        svc.get_pending_orders = MagicMock(
            return_value=[
                {
                    "ticker": "AAPL",
                    "direction": "buy",
                    "quantity": 90000.0,
                    "target_weight": 0.90,
                    "pipeline_run_id": "run-002",
                    "data_version": "data-v2",
                    "feature_version": "feat-v2",
                    "model_version": "model-v2",
                }
            ]
        )

        snapshot = svc.build_pretrade_snapshot("2024-01-15")

        assert snapshot["pretrade"]["blocked"] is True
        assert any(
            breach["limit_name"] == "Position Limit"
            for breach in snapshot["pretrade"]["breaches"]
        )

    def test_build_pretrade_snapshot_allows_initial_bootstrap_turnover(self, monkeypatch):
        monkeypatch.setenv("MLCOUNCIL_AUTOMATION_PAUSED", "false")
        monkeypatch.setenv("MLCOUNCIL_MAX_TURNOVER", "0.30")

        svc = _make_service(account={"portfolio_value": 100000.0, "buying_power": 100000.0})
        svc.get_pending_orders = MagicMock(
            return_value=[
                {
                    "ticker": "AAPL",
                    "direction": "buy",
                    "quantity": 10000.0,
                    "target_weight": 0.10,
                    "pipeline_run_id": "run-bootstrap",
                    "data_version": "data-v1",
                    "feature_version": "feat-v1",
                    "model_version": "model-v1",
                },
                {
                    "ticker": "MSFT",
                    "direction": "buy",
                    "quantity": 10000.0,
                    "target_weight": 0.10,
                    "pipeline_run_id": "run-bootstrap",
                    "data_version": "data-v1",
                    "feature_version": "feat-v1",
                    "model_version": "model-v1",
                },
                {
                    "ticker": "GOOGL",
                    "direction": "buy",
                    "quantity": 10000.0,
                    "target_weight": 0.10,
                    "pipeline_run_id": "run-bootstrap",
                    "data_version": "data-v1",
                    "feature_version": "feat-v1",
                    "model_version": "model-v1",
                },
                {
                    "ticker": "AMZN",
                    "direction": "buy",
                    "quantity": 10000.0,
                    "target_weight": 0.10,
                    "pipeline_run_id": "run-bootstrap",
                    "data_version": "data-v1",
                    "feature_version": "feat-v1",
                    "model_version": "model-v1",
                },
            ]
        )

        snapshot = svc.build_pretrade_snapshot("2024-01-15")

        assert snapshot["reconciliation"]["current_symbols"] == []
        assert snapshot["pretrade"]["projected_turnover"] > 0.30
        assert snapshot["pretrade"]["blocked"] is False
        assert any("bootstrap" in warning.lower() for warning in snapshot["pretrade"]["warnings"])

    def test_execute_orders_converts_notional_to_share_quantity(self, monkeypatch):
        monkeypatch.setenv("MLCOUNCIL_AUTOMATION_PAUSED", "false")
        svc = _make_service()
        svc.get_pending_orders = MagicMock(
            return_value=[
                {
                    "ticker": "AAPL",
                    "direction": "buy",
                    "quantity": 1000.0,
                    "target_weight": 0.01,
                    "pipeline_run_id": "run-003",
                    "data_version": "data-v3",
                    "feature_version": "feat-v3",
                    "model_version": "model-v3",
                }
            ]
        )
        # Isolate from stale repo-state execution records
        svc._load_execution_record = MagicMock(return_value=None)
        svc._execution_record_path = MagicMock(return_value=Path("/tmp/nonexistent"))

        result = svc.execute_orders("2024-01-99")

        svc._node.submit_order.assert_called_once_with("AAPL", 5, "buy")
        assert result["orders_submitted"] == 1
        assert result["pretrade"]["blocked"] is False
        assert result["lineage"]["pipeline_run_id"] == "run-003"

    def test_execute_orders_returns_lineage_reconciliation_and_operations_log(self, monkeypatch):
        from api.services import trading_service as ts
        from council import risk_engine as risk_mod

        monkeypatch.setenv("MLCOUNCIL_AUTOMATION_PAUSED", "false")

        positions = pd.DataFrame(
            [
                {
                    "symbol": "MSFT",
                    "qty": 4,
                    "avg_price": 250.0,
                    "current_price": 250.0,
                    "current_value": 1000.0,
                }
            ]
        )
        svc = _make_service(positions=positions)
        svc.get_pending_orders = MagicMock(
            return_value=[
                {
                    "ticker": "AAPL",
                    "direction": "buy",
                    "quantity": 1000.0,
                    "target_weight": 0.01,
                    "pipeline_run_id": "run-004",
                    "data_version": "data-v4",
                    "feature_version": "feat-v4",
                    "model_version": "model-v4",
                }
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            original_trade_dir = ts.TRADE_LOG_DIR
            original_ops_dir = ts.OPERATIONS_DIR
            original_risk_dir = risk_mod.RISK_DIR
            ts.TRADE_LOG_DIR = tmp_path / "paper_trades"
            ts.OPERATIONS_DIR = tmp_path / "operations"
            risk_mod.RISK_DIR = tmp_path / "risk"
            try:
                result = svc.execute_orders("2024-01-15")
            finally:
                ts.TRADE_LOG_DIR = original_trade_dir
                ts.OPERATIONS_DIR = original_ops_dir
                risk_mod.RISK_DIR = original_risk_dir

            operations_path = Path(result["operations_path"])
            assert operations_path.exists()
            payload = json.loads(operations_path.read_text())

        assert result["lineage"] == {
            "pipeline_run_id": "run-004",
            "data_version": "data-v4",
            "feature_version": "feat-v4",
            "model_version": "model-v4",
        }
        assert result["reconciliation"]["symbols_to_close"] == ["MSFT"]
        assert result["reconciliation"]["symbols_to_open"] == ["AAPL"]
        assert payload["trade_status"] == "success"
        assert payload["orders_submitted"] == 2

    def test_execute_orders_is_idempotent_when_execution_record_exists(self, monkeypatch):
        from api.services import trading_service as ts

        monkeypatch.setenv("MLCOUNCIL_AUTOMATION_PAUSED", "false")

        svc = _make_service()

        with tempfile.TemporaryDirectory() as tmpdir:
            ops_dir = Path(tmpdir) / "operations"
            ops_dir.mkdir(parents=True, exist_ok=True)
            execution_path = ops_dir / "2024-01-15_execution.json"
            execution_path.write_text(
                json.dumps(
                    {
                        "date": "2024-01-15",
                        "pretrade": {"blocked": False},
                        "reconciliation": {"symbols_to_open": ["AAPL"]},
                        "lineage": {"pipeline_run_id": "run-005"},
                    }
                )
            )

            original_ops_dir = ts.OPERATIONS_DIR
            ts.OPERATIONS_DIR = ops_dir
            try:
                result = svc.execute_orders("2024-01-15")
            finally:
                ts.OPERATIONS_DIR = original_ops_dir

        assert result["error"] == "Orders for 2024-01-15 have already been executed"
        svc._node.submit_order.assert_not_called()

    def test_execute_orders_writes_execution_record_after_success(self, monkeypatch):
        from api.services import trading_service as ts
        from council import risk_engine as risk_mod

        monkeypatch.setenv("MLCOUNCIL_AUTOMATION_PAUSED", "false")

        svc = _make_service()
        svc.get_pending_orders = MagicMock(
            return_value=[
                {
                    "ticker": "AAPL",
                    "direction": "buy",
                    "quantity": 1000.0,
                    "target_weight": 0.01,
                    "pipeline_run_id": "run-006",
                    "data_version": "data-v6",
                    "feature_version": "feat-v6",
                    "model_version": "model-v6",
                }
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            original_trade_dir = ts.TRADE_LOG_DIR
            original_ops_dir = ts.OPERATIONS_DIR
            original_risk_dir = risk_mod.RISK_DIR
            ts.TRADE_LOG_DIR = tmp_path / "paper_trades"
            ts.OPERATIONS_DIR = tmp_path / "operations"
            risk_mod.RISK_DIR = tmp_path / "risk"
            try:
                result = svc.execute_orders("2024-01-15")
            finally:
                ts.TRADE_LOG_DIR = original_trade_dir
                ts.OPERATIONS_DIR = original_ops_dir
                risk_mod.RISK_DIR = original_risk_dir

            execution_path = tmp_path / "operations" / "2024-01-15_execution.json"
            assert execution_path.exists()
            payload = json.loads(execution_path.read_text())

        assert result["orders_submitted"] == 1
        assert payload["trade_status"] == "success"
        assert payload["lineage"]["pipeline_run_id"] == "run-006"

    def test_execute_intraday_decision_converts_execution_intents_and_keeps_lineage(self, monkeypatch):
        from api.services import trading_service as ts

        monkeypatch.setenv("MLCOUNCIL_AUTOMATION_PAUSED", "false")

        svc = _make_service()
        decision = {
            "decision_id": "decision-789-test",
            "as_of": "2026-04-09T14:45:00+00:00",
            "strategy_version": "intraday-v1",
            "market_session": "regular",
            "execution_intents": [
                {
                    "ticker": "AAPL",
                    "side": "buy",
                    "target_weight": 0.01,
                    "quantity_notional": 1000.0,
                    "confidence": 0.81,
                    "rationale": "AAPL strongest intraday long",
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            original_ops_dir = ts.OPERATIONS_DIR
            original_intraday_ops_dir = ts.INTRADAY_OPERATIONS_DIR
            ts.OPERATIONS_DIR = tmp_path / "operations"
            ts.INTRADAY_OPERATIONS_DIR = ts.OPERATIONS_DIR / "intraday"
            try:
                result = svc.execute_intraday_decision(decision)
            finally:
                ts.OPERATIONS_DIR = original_ops_dir
                ts.INTRADAY_OPERATIONS_DIR = original_intraday_ops_dir

        svc._node.submit_order.assert_called_once_with("AAPL", 5, "buy")
        assert result["orders_submitted"] == 1
        assert result["lineage"]["decision_id"] == "decision-789-test"
        assert result["lineage"]["strategy_version"] == "intraday-v1"

    def test_prepare_execution_context_honors_explicit_symbols_to_close(self, monkeypatch):
        svc = _make_service()
        monkeypatch.setenv("MLCOUNCIL_AUTOMATION_PAUSED", "false")

        captured: dict[str, object] = {}

        def fake_compute_projected_risk(**kwargs):
            captured.update(kwargs)
            return {
                "path": "risk.json",
                "breaches": [],
                "warnings": [],
                "blocked": False,
                "reason": None,
            }

        svc._compute_projected_risk = fake_compute_projected_risk

        svc._prepare_execution_context_from_orders(
            execution_key="2026-04-09",
            orders=[
                {
                    "ticker": "AAPL",
                    "direction": "buy",
                    "quantity": 1000.0,
                    "target_weight": 0.01,
                }
            ],
            symbols_to_close=["BTCUSD"],
        )

        assert captured["symbols_to_close"] == ["BTCUSD"]

    def test_execute_intraday_decision_is_idempotent_per_decision_id(self, monkeypatch):
        from api.services import trading_service as ts

        monkeypatch.setenv("MLCOUNCIL_AUTOMATION_PAUSED", "false")
        svc = _make_service()
        decision = {
            "decision_id": "decision-repeat",
            "as_of": "2026-04-09T14:45:00+00:00",
            "strategy_version": "intraday-v1",
            "market_session": "regular",
            "execution_intents": [
                {
                    "ticker": "AAPL",
                    "side": "buy",
                    "target_weight": 0.01,
                    "quantity_notional": 1000.0,
                    "confidence": 0.81,
                    "rationale": "AAPL strongest intraday long",
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            original_ops_dir = ts.OPERATIONS_DIR
            original_intraday_ops_dir = ts.INTRADAY_OPERATIONS_DIR
            ts.OPERATIONS_DIR = tmp_path / "operations"
            ts.INTRADAY_OPERATIONS_DIR = ts.OPERATIONS_DIR / "intraday"
            try:
                first = svc.execute_intraday_decision(decision)
                second = svc.execute_intraday_decision(decision)
            finally:
                ts.OPERATIONS_DIR = original_ops_dir
                ts.INTRADAY_OPERATIONS_DIR = original_intraday_ops_dir

        assert first["orders_submitted"] == 1
        assert "already been executed" in second["error"]
        svc._node.submit_order.assert_called_once_with("AAPL", 5, "buy")

    def test_execute_intraday_decision_preserves_fractional_crypto_quantity(self, monkeypatch):
        from api.services import trading_service as ts
        from council import risk_engine as risk_mod

        monkeypatch.setenv("MLCOUNCIL_AUTOMATION_PAUSED", "false")
        svc = _make_service()
        svc._node.get_latest_price.return_value = 70000.0
        decision = {
            "decision_id": "decision-btc-fractional",
            "as_of": "2026-04-13T09:15:00+00:00",
            "strategy_version": "intraday-v1",
            "market_session": "crypto",
            "execution_intents": [
                {
                    "ticker": "BTCUSD",
                    "side": "sell",
                    "target_weight": 0.055,
                    "quantity_notional": 5500.0,
                    "confidence": 0.45,
                    "rationale": "BTCUSD weakest intraday asset",
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            original_ops_dir = ts.OPERATIONS_DIR
            original_intraday_ops_dir = ts.INTRADAY_OPERATIONS_DIR
            original_risk_dir = risk_mod.RISK_DIR
            ts.OPERATIONS_DIR = tmp_path / "operations"
            ts.INTRADAY_OPERATIONS_DIR = ts.OPERATIONS_DIR / "intraday"
            risk_mod.RISK_DIR = tmp_path / "risk"
            try:
                result = svc.execute_intraday_decision(decision)
            finally:
                ts.OPERATIONS_DIR = original_ops_dir
                ts.INTRADAY_OPERATIONS_DIR = original_intraday_ops_dir
                risk_mod.RISK_DIR = original_risk_dir

        submitted_qty = svc._node.submit_order.call_args[0][1]
        assert submitted_qty == pytest.approx(5500.0 / 70000.0)
        assert submitted_qty > 0
        assert result["orders_submitted"] == 1
