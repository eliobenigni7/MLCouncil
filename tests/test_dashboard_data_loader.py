from __future__ import annotations

import importlib
import json
import sys
import types
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import pytest


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


def test_load_equity_curve_densifies_sparse_business_days(monkeypatch, tmp_path):
    data_loader = _load_data_loader(monkeypatch)
    _configure_loader_paths(monkeypatch, data_loader, tmp_path)

    _write_risk_report(tmp_path, "2026-04-08", 100_000.0)
    _write_risk_report(tmp_path, "2026-04-10", 110_000.0)

    equity = data_loader.load_equity_curve("Paper Trading")

    assert list(equity.index.strftime("%Y-%m-%d")) == ["2026-04-08", "2026-04-09", "2026-04-10"]
    assert equity.round(2).tolist() == [100.0, 100.0, 110.0]


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


# ============================================================================
# Loaders used by the dashboard pages (F-0.3) — real fixtures in tmp_path
# ============================================================================

def _write_orders_snapshot(
    root: Path,
    as_of: str,
    portfolio_value: float,
    *,
    regime: str | None = None,
) -> None:
    """Write a minimal daily orders parquet under tmp data/orders."""
    orders_dir = root / "data" / "orders"
    orders_dir.mkdir(parents=True, exist_ok=True)
    row: dict = {"ticker": "AAPL", "portfolio_value": portfolio_value}
    if regime is not None:
        row.update(
            {
                "regime": regime,
                "prob_bull": 0.7,
                "prob_bear": 0.2,
                "prob_transition": 0.1,
            }
        )
    pd.DataFrame([row]).to_parquet(orders_dir / f"{as_of}.parquet")


def _write_attribution(root: Path, df: pd.DataFrame) -> None:
    results_dir = root / "data" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(results_dir / "attribution.parquet")


def test_load_sidebar_metrics_aggregates_real_artifacts(monkeypatch, tmp_path):
    data_loader = _load_data_loader(monkeypatch)
    _configure_loader_paths(monkeypatch, data_loader, tmp_path)

    _write_orders_snapshot(tmp_path, "2026-04-08", 100_000.0, regime="bull")
    _write_orders_snapshot(tmp_path, "2026-04-09", 105_000.0, regime="bull")
    _write_orders_snapshot(tmp_path, "2026-04-10", 110_000.0, regime="bull")

    _write_attribution(
        tmp_path,
        pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-04-08", "2026-04-09", "2026-04-10"]),
                "model_name": ["lgbm", "lgbm", "lgbm"],
                "weight": [0.5, 0.5, 0.5],
                "ic_rolling_30d": [0.04, 0.05, 0.06],
                "sharpe_rolling_60d": [1.2, 1.2, 1.2],
                "pnl_contribution": [0.01, 0.01, 0.01],
            }
        ),
    )

    metrics = data_loader.load_sidebar_metrics()

    assert metrics["regime"] == "Bull"
    assert metrics["regime_prob"] == 70.0
    assert metrics["sharpe_ytd"] > 0
    assert metrics["max_dd"] == 0.0  # monotonically increasing equity
    assert metrics["ic_30d"] == pytest.approx(0.06)


def test_load_ic_history_and_weights_history_pivot_attribution_parquet(
    monkeypatch, tmp_path
):
    data_loader = _load_data_loader(monkeypatch)
    _configure_loader_paths(monkeypatch, data_loader, tmp_path)

    rows = []
    for d, ic, w in [("2026-04-08", 0.04, 0.5), ("2026-04-09", 0.05, 0.4)]:
        rows.append({"date": pd.Timestamp(d), "model_name": "lgbm", "weight": w,
                     "ic_rolling_30d": ic, "sharpe_rolling_60d": 1.0,
                     "pnl_contribution": 0.01})
        rows.append({"date": pd.Timestamp(d), "model_name": "sentiment", "weight": 0.3,
                     "ic_rolling_30d": ic * 0.6, "sharpe_rolling_60d": 0.9,
                     "pnl_contribution": 0.005})
        rows.append({"date": pd.Timestamp(d), "model_name": "hmm", "weight": 0.2,
                     "ic_rolling_30d": ic * 0.4, "sharpe_rolling_60d": 0.8,
                     "pnl_contribution": 0.003})
    _write_attribution(tmp_path, pd.DataFrame(rows))

    ic_history = data_loader.load_ic_history()
    weights_history = data_loader.load_weights_history()

    assert set(ic_history.columns) == {"date", "lgbm", "sentiment", "hmm"}
    assert ic_history.iloc[0]["lgbm"] == pytest.approx(0.04)
    assert ic_history.iloc[1]["hmm"] == pytest.approx(0.02)

    assert set(weights_history.columns) == {"date", "lgbm", "sentiment", "hmm"}
    assert weights_history.iloc[0][["lgbm", "sentiment", "hmm"]].sum() == pytest.approx(1.0)
    assert weights_history.iloc[1]["lgbm"] == pytest.approx(0.4)


def test_load_current_regime_from_orders_parquet(monkeypatch, tmp_path):
    data_loader = _load_data_loader(monkeypatch)
    _configure_loader_paths(monkeypatch, data_loader, tmp_path)

    _write_orders_snapshot(tmp_path, "2026-04-08", 100_000.0, regime="transition")
    _write_orders_snapshot(tmp_path, "2026-04-09", 105_000.0, regime="bull")

    regime = data_loader.load_current_regime()

    assert regime == {
        "regime": "bull",
        "bull": 0.7,
        "bear": 0.2,
        "transition": 0.1,
    }


def test_load_optimization_diagnostics_reads_json_and_missing(monkeypatch, tmp_path):
    data_loader = _load_data_loader(monkeypatch)
    _configure_loader_paths(monkeypatch, data_loader, tmp_path)

    diag_dir = tmp_path / "data" / "results" / "optimization_diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    payload = {"solver_status": "optimal", "greedy_weights": {"AAPL": 0.5}}
    (diag_dir / "2026-04-10.json").write_text(json.dumps(payload), encoding="utf-8")

    assert data_loader.load_optimization_diagnostics(date(2026, 4, 10)) == payload
    assert data_loader.load_optimization_diagnostics(date(2026, 4, 11)) == {}


def test_load_council_weights_log_entry_empty_without_aggregator_pkl(
    monkeypatch, tmp_path
):
    data_loader = _load_data_loader(monkeypatch)
    _configure_loader_paths(monkeypatch, data_loader, tmp_path)

    assert data_loader.load_council_weights_log_entry(date(2026, 4, 10)) == {}


def test_load_fill_quality_summary_from_fill_log_and_calibration(monkeypatch, tmp_path):
    data_loader = _load_data_loader(monkeypatch)
    _configure_loader_paths(monkeypatch, data_loader, tmp_path)

    # 1. Real fill log partition (same writer the pipeline uses)
    from execution.fill_log import FillRecord, append_fills

    fills_dir = tmp_path / "data" / "operations" / "fills"
    append_fills(
        [
            FillRecord(
                fill_id="f1", order_id="o1", ticker="AAPL", side="buy", qty=10,
                fill_price=100.02, decision_price=100.0,
                decision_ts=datetime(2026, 4, 8, 13, 0, tzinfo=timezone.utc),
                fill_ts=datetime(2026, 4, 8, 13, 5, tzinfo=timezone.utc),
            ),
            FillRecord(
                fill_id="f2", order_id="o2", ticker="MSFT", side="sell", qty=5,
                fill_price=99.95, decision_price=100.0,
                decision_ts=datetime(2026, 4, 8, 13, 0, tzinfo=timezone.utc),
                fill_ts=datetime(2026, 4, 8, 13, 6, tzinfo=timezone.utc),
            ),
        ],
        base=fills_dir,
    )

    # 2. Calibration artifact with matching .manifest sidecar
    import council.cost_calibration as cost_calibration
    import council.transaction_costs as transaction_costs

    calib = {
        "generated_at": "2026-04-10T00:00:00+00:00",
        "calibration_window_end": "2026-04-10T00:00:00+00:00",
        "fill_sample_count": 2,
        "min_fills": 1,
        "kappa_by_ticker": {"AAPL": 1.5, "MSFT": 2.0},
        "fill_count_by_ticker": {"AAPL": 1, "MSFT": 1},
        "kappa_by_tier": {},
        "fill_count_by_tier": {},
    }
    payload = json.dumps(calib).encode("utf-8")
    calib_path = tmp_path / "data" / "operations" / "cost_calibration.json"
    calib_path.write_bytes(payload)
    (calib_path.with_suffix(".json.manifest")).write_text(
        json.dumps({"sha256": cost_calibration._calibration_version(payload)}),
        encoding="utf-8",
    )
    monkeypatch.setattr(transaction_costs, "get_calibration_path", lambda: calib_path)

    summary = data_loader.load_fill_quality_summary()
    by_ticker = summary.set_index("ticker")

    assert set(by_ticker.index) == {"AAPL", "MSFT"}
    # IS for the buy: 10_000 * (100.02 - 100.0) / 100.0 = 2.0 bps
    assert by_ticker.loc["AAPL", "median_is_bps"] == pytest.approx(2.0)
    assert by_ticker.loc["MSFT", "median_is_bps"] == pytest.approx(5.0)
    assert by_ticker.loc["AAPL", "fill_count"] == 1
    assert by_ticker.loc["AAPL", "kappa_calibrated_bps"] == pytest.approx(1.5)
    assert by_ticker.loc["AAPL", "lookup_slippage_bps"] == pytest.approx(2.0)


def test_load_fill_quality_summary_empty_without_fill_log(monkeypatch, tmp_path):
    data_loader = _load_data_loader(monkeypatch)
    _configure_loader_paths(monkeypatch, data_loader, tmp_path)

    assert data_loader.load_fill_quality_summary().empty
