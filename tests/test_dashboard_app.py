"""Entry-point smoke tests for the Streamlit dashboard (roadmap F-0.3).

Strategy: ``streamlit.testing.v1.AppTest`` is available in this environment, so
each entry point (``dashboard/app.py`` + the three pages) is executed end-to-end
with the data loaders monkeypatched to serve minimal fake artifacts. This
covers both "module imports without errors" and "main function runs the happy
path without crashing", exercising the full top-level bodies of the scripts.
No dashboard source file is modified; design/layout is untouched.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import backtest.playground as playground  # noqa: E402
import council.alerts as alerts  # noqa: E402
import council.cost_calibration as cost_calibration  # noqa: E402
import council.transaction_costs as transaction_costs  # noqa: E402
import dashboard.data_loader as data_loader  # noqa: E402

_DASHBOARD_DIR = _ROOT / "dashboard"

_ATTRIBUTION_COLUMNS = [
    "date",
    "model_name",
    "weight",
    "ic_rolling_30d",
    "sharpe_rolling_60d",
    "pnl_contribution",
]
_REGIME_COLUMNS = ["date", "regime", "prob_bull", "prob_bear", "prob_transition"]


def _run_entry_point(name: str):
    from streamlit.testing.v1 import AppTest

    return AppTest.from_file(str(_DASHBOARD_DIR / name), default_timeout=60)


# ============================================================================
# Minimal fake artifacts
# ============================================================================

def _fake_equity(n: int = 60) -> pd.Series:
    idx = pd.bdate_range(end=pd.Timestamp("2026-04-10"), periods=n)
    vals = 100.0 * np.cumprod(1 + np.random.default_rng(0).normal(0.0004, 0.01, n))
    return pd.Series(vals, index=idx, name="equity_normalized")


def _fake_attribution() -> pd.DataFrame:
    dates = pd.to_datetime(["2026-04-08", "2026-04-09", "2026-04-10"])
    rows = []
    for d in dates:
        rows.append({"date": d, "model_name": "lgbm", "weight": 0.5, "ic_rolling_30d": 0.05,
                     "sharpe_rolling_60d": 1.2, "pnl_contribution": 0.02})
        rows.append({"date": d, "model_name": "sentiment", "weight": 0.3, "ic_rolling_30d": 0.03,
                     "sharpe_rolling_60d": 1.0, "pnl_contribution": 0.01})
        rows.append({"date": d, "model_name": "hmm", "weight": 0.2, "ic_rolling_30d": 0.02,
                     "sharpe_rolling_60d": 0.9, "pnl_contribution": 0.005})
    return pd.DataFrame(rows)


def _fake_regime_history() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-04-08", "2026-04-09", "2026-04-10"]),
            "regime": ["bull", "bull", "transition"],
            "prob_bull": [0.7, 0.7, 0.3],
            "prob_bear": [0.2, 0.2, 0.2],
            "prob_transition": [0.1, 0.1, 0.5],
        }
    )


def _empty_sidebar_metrics() -> dict:
    return {
        "sharpe_ytd": 0.0, "max_dd": 0.0, "ic_30d": 0.0,
        "regime": "N/A", "regime_prob": 0.0,
        "sharpe_delta": 0.0, "dd_delta": 0.0, "ic_delta": 0.0,
    }


# ============================================================================
# Loader patches
# ============================================================================

def _patch_app_loaders(monkeypatch, *, empty: bool = False) -> None:
    """Monkeypatch every data loader used by ``dashboard/app.py``.

    With ``empty=True`` the fake artifacts are empty states, exercising the
    no-data code paths of all three tabs.
    """
    if empty:
        equity = pd.Series(dtype=float, name="equity_normalized")
        returns = pd.Series(dtype=float, name="returns")
        attribution = pd.DataFrame(columns=_ATTRIBUTION_COLUMNS)
        regime_info = {"regime": "unknown", "bull": 0.0, "bear": 0.0, "transition": 0.0}
        regime_history = pd.DataFrame(columns=_REGIME_COLUMNS)
        sidebar = _empty_sidebar_metrics()
    else:
        equity = _fake_equity()
        returns = equity.pct_change().dropna()
        attribution = _fake_attribution()
        regime_info = {"regime": "bull", "bull": 0.7, "bear": 0.2, "transition": 0.1}
        regime_history = _fake_regime_history()
        sidebar = {
            "sharpe_ytd": 1.2, "max_dd": -8.5, "ic_30d": 0.042, "regime": "Bull",
            "regime_prob": 72.0, "sharpe_delta": 0.1, "dd_delta": 0.0, "ic_delta": 0.01,
        }

    if attribution.empty:
        ic_history = pd.DataFrame()
        weights_history = pd.DataFrame()
    else:
        ic_history = (
            attribution[["date", "model_name", "ic_rolling_30d"]]
            .pivot_table(index="date", columns="model_name", values="ic_rolling_30d")
            .reset_index()
        )
        weights_history = (
            attribution[["date", "model_name", "weight"]]
            .pivot_table(index="date", columns="model_name", values="weight")
            .reset_index()
        )

    monkeypatch.setattr(
        data_loader, "load_equity_curve",
        lambda mode="Paper Trading", results_tag=None: equity.copy(),
    )
    monkeypatch.setattr(
        data_loader, "load_benchmark",
        lambda mode="Paper Trading", results_tag=None: equity.copy() * 1.02,
    )
    monkeypatch.setattr(
        data_loader, "load_daily_returns",
        lambda mode="Paper Trading", results_tag=None: returns.copy(),
    )
    monkeypatch.setattr(
        data_loader, "load_model_attribution",
        lambda start=None, end=None: attribution.copy(),
    )
    monkeypatch.setattr(data_loader, "load_ic_history", lambda: ic_history.copy())
    monkeypatch.setattr(data_loader, "load_weights_history", lambda: weights_history.copy())
    monkeypatch.setattr(data_loader, "load_current_regime", lambda: dict(regime_info))
    monkeypatch.setattr(data_loader, "load_regime_history", lambda: regime_history.copy())
    monkeypatch.setattr(data_loader, "load_sidebar_metrics", lambda: dict(sidebar))
    monkeypatch.setattr(data_loader, "load_council_weights_log_entry", lambda as_of: {})
    monkeypatch.setattr(data_loader, "load_optimization_diagnostics", lambda as_of: {})
    monkeypatch.setattr(alerts, "load_current_alerts", lambda: [])


def _patch_fill_quality_page(monkeypatch, tmp_path: Path, summary: pd.DataFrame) -> None:
    """Patch the disk-facing pieces used by ``pages/1_Fill_Quality.py``."""
    monkeypatch.setattr(data_loader, "load_fill_quality_summary", lambda: summary.copy())
    monkeypatch.setattr(transaction_costs, "get_calibration_path", lambda: None)
    monkeypatch.setattr(transaction_costs, "get_active_calibration_version", lambda path=None: "")
    # Calibration artifact path -> nonexistent file so the page hits the
    # "not produced yet" branch deterministically.
    monkeypatch.setattr(
        cost_calibration, "DEFAULT_CALIBRATION_PATH",
        tmp_path / "operations" / "_nonexistent_calibration.json",
    )


def _patch_playground_page(monkeypatch) -> None:
    """Patch the disk-facing pieces used by ``pages/3_Backtest_Playground.py``."""
    monkeypatch.setattr(playground, "load_available_universe", lambda: ["AAPL", "MSFT", "GOOGL"])
    monkeypatch.setattr(
        playground, "list_snapshots",
        lambda base_dir=None: pd.DataFrame(
            columns=["timestamp", "start_date", "end_date", "n_tickers", "sharpe",
                     "max_drawdown", "cagr", "final_equity", "note", "path"]
        ),
    )


# ============================================================================
# dashboard/app.py
# ============================================================================

def test_app_imports_and_runs_happy_path_with_fake_data(monkeypatch):
    _patch_app_loaders(monkeypatch, empty=False)

    at = _run_entry_point("app.py").run()

    assert not at.exception
    assert len(at.get("metric")) >= 4  # sidebar live metrics row
    assert len(at.get("plotly_chart")) >= 5  # equity, sharpe, dd, heatmap, radar...


def test_app_renders_without_crash_when_no_data(monkeypatch):
    _patch_app_loaders(monkeypatch, empty=True)

    at = _run_entry_point("app.py").run()

    assert not at.exception
    assert len(at.get("metric")) >= 4  # sidebar metrics with zeroed values


# ============================================================================
# dashboard/pages/1_Fill_Quality.py
# ============================================================================

def test_fill_quality_page_imports_and_runs_with_fills(monkeypatch, tmp_path):
    summary = pd.DataFrame(
        {
            "ticker": ["AAPL", "MSFT"],
            "median_is_bps": [1.2, 0.8],
            "fill_count": [12, 9],
            "lookup_slippage_bps": [3.0, 3.0],
            "kappa_calibrated_bps": [1.5, 0.9],
        }
    )
    _patch_fill_quality_page(monkeypatch, tmp_path, summary)

    at = _run_entry_point("pages/1_Fill_Quality.py").run()

    assert not at.exception
    assert len(at.get("dataframe")) >= 1  # fill summary table


def test_fill_quality_page_handles_empty_fill_log(monkeypatch, tmp_path):
    _patch_fill_quality_page(monkeypatch, tmp_path, pd.DataFrame())

    at = _run_entry_point("pages/1_Fill_Quality.py").run()

    assert not at.exception
    assert len(at.info) >= 1  # "No fill records ..." + no calibration artifact


# ============================================================================
# dashboard/pages/2_Challenger_Promotion.py
# ============================================================================

def test_challenger_promotion_page_imports_and_runs(monkeypatch):
    # Reads config/production_manifest.yaml and data/operations/*.json directly
    # from disk; every read is guarded (missing/malformed -> skipped), so the
    # page is safe against both present and absent artifacts.
    at = _run_entry_point("pages/2_Challenger_Promotion.py").run()

    assert not at.exception
    assert len(at.get("dataframe")) >= 1  # walk-forward status table (4 models)


# ============================================================================
# dashboard/pages/3_Backtest_Playground.py
# ============================================================================

def test_backtest_playground_page_imports_and_runs_empty(monkeypatch):
    _patch_playground_page(monkeypatch)

    at = _run_entry_point("pages/3_Backtest_Playground.py").run()

    assert not at.exception
    assert len(at.info) >= 1  # "set parameters..." + "no snapshots yet"
    assert len(at.button) >= 1  # "Run Backtest" button


def test_backtest_playground_run_click_submits_and_renders_result(monkeypatch):
    _patch_playground_page(monkeypatch)

    def fake_run(params, cb):
        idx = pd.bdate_range("2026-01-05", "2026-02-27")
        equity = pd.Series(100.0 * (1.001 ** np.arange(len(idx))), index=idx, name="equity")
        return playground.PlaygroundResult(
            params=params,
            equity_curve=equity,
            gross_equity_curve=equity,
            benchmark_curve=equity * 1.01,
            weights=pd.DataFrame({"lgbm": [0.5], "sentiment": [0.3], "hmm": [0.2]}),
            council_contributions=pd.DataFrame(
                {"contrib_lgbm": [0.01], "contrib_sentiment": [0.005], "contrib_hmm": [0.003]}
            ),
            stats={
                "sharpe": 1.5, "max_drawdown": -0.05, "cagr": 0.2, "turnover": 0.1,
                "calmar": 1.2, "final_equity": 110000.0, "estimated_costs_usd": 150.0,
                "n_trades": 12,
            },
            snapshot_path=None,
        )

    monkeypatch.setattr(playground, "run_playground_backtest", fake_run)

    at = _run_entry_point("pages/3_Backtest_Playground.py").run()
    at.button[0].click().run()

    assert not at.exception
    assert at.session_state["playground_last_error"] is None
    assert at.session_state["playground_last_result"] is not None
    assert len(at.get("metric")) >= 8  # _render_stats: 4 + 4 metrics
