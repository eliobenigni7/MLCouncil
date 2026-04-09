from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path

import pandas as pd


def _load_data_loader(monkeypatch):
    streamlit = types.ModuleType("streamlit")

    def cache_data(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    streamlit.cache_data = cache_data
    monkeypatch.setitem(sys.modules, "streamlit", streamlit)
    sys.modules.pop("dashboard.data_loader", None)

    import dashboard.data_loader as data_loader

    return importlib.reload(data_loader)


def _configure_loader_paths(monkeypatch, data_loader, root: Path) -> None:
    monkeypatch.setattr(data_loader, "_ROOT", root)
    monkeypatch.setattr(data_loader, "_ORDERS_DIR", root / "data" / "orders")
    monkeypatch.setattr(data_loader, "_RAW_DIR", root / "data" / "raw")
    monkeypatch.setattr(data_loader, "_RESULTS_DIR", root / "data" / "results")
    monkeypatch.setattr(data_loader, "_RISK_DIR", root / "data" / "risk", raising=False)
    monkeypatch.setattr(data_loader, "_PAPER_TRADES_DIR", root / "data" / "paper_trades", raising=False)
    monkeypatch.setattr(data_loader, "_OPERATIONS_DIR", root / "data" / "operations", raising=False)


def _write_risk_report(root: Path, as_of: str, portfolio_value: float) -> None:
    risk_dir = root / "data" / "risk"
    risk_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": f"{as_of}T16:00:00+00:00",
        "portfolio_value": portfolio_value,
        "var": {},
        "exposure": {},
        "pnl_today": 0.0,
        "return_today": 0.0,
        "volatility_1d": 0.0,
        "volatility_20d": 0.0,
        "sharpe_estimate": 0.0,
        "max_drawdown_current": 0.0,
        "breaches": [],
    }
    (risk_dir / f"risk_report_{as_of}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_load_equity_curve_prefers_real_risk_reports_when_orders_lack_portfolio_value(
    monkeypatch,
    tmp_path,
):
    data_loader = _load_data_loader(monkeypatch)
    _configure_loader_paths(monkeypatch, data_loader, tmp_path)

    orders_dir = tmp_path / "data" / "orders"
    orders_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"ticker": "AAPL", "direction": "buy", "quantity": 10, "target_weight": 0.1}]
    ).to_parquet(orders_dir / "2026-04-08.parquet")

    _write_risk_report(tmp_path, "2026-04-08", 100_000.0)
    _write_risk_report(tmp_path, "2026-04-09", 105_000.0)

    equity = data_loader.load_equity_curve("Paper Trading")

    assert list(equity.index.strftime("%Y-%m-%d")) == ["2026-04-08", "2026-04-09"]
    assert equity.round(2).tolist() == [100.0, 105.0]


def test_load_benchmark_uses_real_sp500_macro_series(monkeypatch, tmp_path):
    data_loader = _load_data_loader(monkeypatch)
    _configure_loader_paths(monkeypatch, data_loader, tmp_path)

    _write_risk_report(tmp_path, "2026-04-08", 100_000.0)
    _write_risk_report(tmp_path, "2026-04-09", 105_000.0)

    macro_dir = tmp_path / "data" / "raw" / "macro"
    macro_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "valid_time": pd.to_datetime(["2026-04-08", "2026-04-09"]),
            "sp500_price": [5000.0, 5100.0],
        }
    ).to_parquet(macro_dir / "sp500.parquet")

    benchmark = data_loader.load_benchmark("Paper Trading")

    assert list(benchmark.index.strftime("%Y-%m-%d")) == ["2026-04-08", "2026-04-09"]
    assert benchmark.round(2).tolist() == [100.0, 102.0]


def test_load_model_attribution_returns_empty_without_real_artifacts(monkeypatch, tmp_path):
    data_loader = _load_data_loader(monkeypatch)
    _configure_loader_paths(monkeypatch, data_loader, tmp_path)

    attribution = data_loader.load_model_attribution()

    assert attribution.empty


def test_load_current_regime_returns_unknown_without_real_artifacts(monkeypatch, tmp_path):
    data_loader = _load_data_loader(monkeypatch)
    _configure_loader_paths(monkeypatch, data_loader, tmp_path)

    regime = data_loader.load_current_regime()

    assert regime == {
        "regime": "unknown",
        "bull": 0.0,
        "bear": 0.0,
        "transition": 0.0,
    }


def test_load_regime_history_returns_empty_without_real_artifacts(monkeypatch, tmp_path):
    data_loader = _load_data_loader(monkeypatch)
    _configure_loader_paths(monkeypatch, data_loader, tmp_path)

    history = data_loader.load_regime_history()

    assert history.empty
