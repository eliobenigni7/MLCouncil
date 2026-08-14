"""Parità numerica: analytics_service vs dashboard.data_loader (stessi artifact)."""
from pathlib import Path

import pandas as pd
import pytest

from api.services import analytics_service


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    """Reindirizza data/ in tmp_path per il test."""
    results = tmp_path / "results"
    results.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(analytics_service, "DATA_DIR", tmp_path)
    return tmp_path


def _write_equity_fixture(tmp_path: Path):
    idx = pd.date_range("2024-01-01", "2024-01-10", freq="B")
    equity = pd.Series([100.0, 101.5, 100.2, 102.0, 101.1, 103.0, 102.4, 104.0], index=idx[:8])
    equity.to_frame(name="equity").to_parquet(tmp_path / "results" / "equity_curve.parquet")
    return equity


def test_equity_curve_matches_data_loader(data_dir, tmp_path, monkeypatch):
    import dashboard.data_loader as dl

    # Il repository ha artifact reali (backtest_result.pkl, equity_curve.parquet,
    # ordini paper, risk/paper_trades): reindirizziamo anche data_loader sul
    # fixture, così entrambi i lati leggono lo stesso equity_curve.parquet
    # (branch parquet, niente sidecar pkl/hash).
    monkeypatch.setattr(dl, "_RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(dl, "_ORDERS_DIR", tmp_path / "orders")
    monkeypatch.setattr(dl, "_RISK_DIR", tmp_path / "risk")
    monkeypatch.setattr(dl, "_PAPER_TRADES_DIR", tmp_path / "paper_trades")

    _write_equity_fixture(tmp_path)
    service_out = analytics_service.load_equity_curve(mode="Paper Trading")
    loader_out = dl.load_equity_curve(mode="Paper Trading")

    # data_loader returns the normalized pd.Series; service returns {"dates", "values"}
    assert list(service_out["dates"]) == [d.isoformat() for d in loader_out.index]
    assert service_out["values"] == pytest.approx([float(v) for v in loader_out.values])
