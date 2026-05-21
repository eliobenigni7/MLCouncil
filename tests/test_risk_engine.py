from __future__ import annotations
from datetime import datetime, timezone

import numpy as np
import pandas as pd


def test_risk_engine_flags_position_and_sector_breaches():
    from council.risk_engine import (
        ExposureReport,
        RiskBreach,
        RiskEngine,
        RiskLimits,
        RiskReport,
        VaRReport,
    )

    engine = RiskEngine(
        limits=RiskLimits(
            max_sector_exposure=0.25,
            max_single_position=0.10,
        )
    )

    report = RiskReport(
        timestamp=datetime.now(timezone.utc),
        portfolio_value=100000.0,
        var=VaRReport(
            var_1d=0.0,
            var_5d=0.0,
            var_10d=0.0,
            cvar_1d=0.0,
            cvar_5d=0.0,
            cvar_10d=0.0,
            var_method="historical",
            confidence_level=0.99,
            portfolio_value=100000.0,
        ),
        exposure=ExposureReport(
            total_market_value=100000.0,
            net_exposure=1.0,
            gross_exposure=1.0,
            sector_exposure={"Technology": 90000.0},
            factor_exposure={"beta": 0.1},
            concentration={"AAPL": 0.90},
            beta_exposure=0.1,
            sector_weights={"Technology": 0.90},
        ),
        pnl_today=0.0,
        return_today=0.0,
        volatility_1d=0.0,
        volatility_20d=0.0,
        sharpe_estimate=0.0,
        max_drawdown_current=0.0,
    )

    breaches = engine.check_limits(report)

    assert any(b.limit_name == "Sector Exposure" for b in breaches)
    assert any(b.limit_name == "Position Limit" for b in breaches)


def test_risk_engine_save_and_load_roundtrip(tmp_path):
    from council import risk_engine as risk_mod
    from council.risk_engine import Position, RiskEngine

    engine = RiskEngine()
    returns = pd.DataFrame({"AAPL": [0.01] * 35})
    positions = [
        Position(
            symbol="AAPL",
            quantity=10,
            avg_price=100.0,
            current_price=110.0,
        )
    ]

    original_dir = risk_mod.RISK_DIR
    risk_mod.RISK_DIR = tmp_path
    try:
        report = engine.compute_full_risk(
            positions=positions,
            returns=returns,
            portfolio_value=1100.0,
        )
        path = engine.save_report(report, date="2026-04-08")
        loaded = engine.load_report("2026-04-08")
    finally:
        risk_mod.RISK_DIR = original_dir

    assert path.exists()
    assert loaded is not None
    assert loaded.portfolio_value == 1100.0


def test_monte_carlo_var_is_reproducible_with_seed():
    from council.risk_engine import Position, RiskEngine

    returns = pd.DataFrame(
        {
            "AAPL": [0.01, -0.02, 0.015, -0.01] * 10,
            "MSFT": [0.008, -0.01, 0.012, -0.009] * 10,
        }
    )
    positions = [
        Position(symbol="AAPL", quantity=10, avg_price=100.0, current_price=110.0),
        Position(symbol="MSFT", quantity=8, avg_price=200.0, current_price=210.0),
    ]

    engine = RiskEngine(seed=17)

    report_a = engine.compute_var(
        returns=returns,
        positions=positions,
        portfolio_value=2780.0,
        method="monte_carlo",
    )
    report_b = engine.compute_var(
        returns=returns,
        positions=positions,
        portfolio_value=2780.0,
        method="monte_carlo",
    )

    assert report_a.var_1d == report_b.var_1d
    assert report_a.cvar_1d == report_b.cvar_1d


def test_monte_carlo_var_allows_seed_override():
    from council.risk_engine import Position, RiskEngine

    returns = pd.DataFrame(
        {
            "AAPL": [0.01, -0.02, 0.015, -0.01] * 10,
            "MSFT": [0.008, -0.01, 0.012, -0.009] * 10,
        }
    )
    positions = [
        Position(symbol="AAPL", quantity=10, avg_price=100.0, current_price=110.0),
        Position(symbol="MSFT", quantity=8, avg_price=200.0, current_price=210.0),
    ]

    engine = RiskEngine(seed=17)

    report_a = engine.compute_var(
        returns=returns,
        positions=positions,
        portfolio_value=2780.0,
        method="monte_carlo",
        seed=17,
    )
    report_b = engine.compute_var(
        returns=returns,
        positions=positions,
        portfolio_value=2780.0,
        method="monte_carlo",
        seed=23,
    )

    assert (report_a.var_1d, report_a.cvar_1d) != (report_b.var_1d, report_b.cvar_1d)


def test_monte_carlo_var_reflects_correlation_structure_with_same_marginals():
    from council.risk_engine import RiskEngine

    tickers = ["AAPL", "MSFT", "NVDA"]
    weights = {"AAPL": 0.5, "MSFT": 0.3, "NVDA": 0.2}
    portfolio_value = 1_000_000.0

    std = 0.02
    corr_independent = np.eye(3)
    corr_structured = np.array(
        [
            [1.0, 0.2, -0.3],
            [0.2, 1.0, 0.0],
            [-0.3, 0.0, 1.0],
        ],
        dtype=float,
    )
    cov_scale = (std ** 2)
    cov_independent = corr_independent * cov_scale
    cov_structured = corr_structured * cov_scale

    # Build deterministic return samples with exact sample covariance:
    # sample_cov = X^T X / (n-1) = Sigma
    n_obs = 8
    rng = np.random.default_rng(123)
    base = rng.normal(size=(n_obs, 3))
    base = base - base.mean(axis=0, keepdims=True)
    q, _ = np.linalg.qr(base)
    q = q[:, :3]
    x_independent = np.sqrt(n_obs - 1.0) * q @ np.linalg.cholesky(cov_independent).T
    x_structured = np.sqrt(n_obs - 1.0) * q @ np.linalg.cholesky(cov_structured).T

    returns_independent = pd.DataFrame(x_independent, columns=tickers)
    returns_structured = pd.DataFrame(x_structured, columns=tickers)

    # Marginal vol held constant across scenarios.
    vol_ind = returns_independent.std(ddof=1)
    vol_str = returns_structured.std(ddof=1)
    assert np.allclose(vol_ind.values, vol_str.values, atol=1e-12)

    # Portfolio variance is also held constant here, so univariate MC would
    # generate identical paths under the same seed.
    w = np.array([weights[t] for t in tickers], dtype=float)
    port_var_ind = float(w @ returns_independent.cov().values @ w)
    port_var_str = float(w @ returns_structured.cov().values @ w)
    assert np.isclose(port_var_ind, port_var_str, atol=1e-12)

    engine = RiskEngine(seed=7)
    var_ind = engine.compute_var_monte_carlo(
        returns=returns_independent,
        weights=weights,
        portfolio_value=portfolio_value,
        n_simulations=4000,
        confidence=0.99,
        horizon=1,
    )
    var_str = engine.compute_var_monte_carlo(
        returns=returns_structured,
        weights=weights,
        portfolio_value=portfolio_value,
        n_simulations=4000,
        confidence=0.99,
        horizon=1,
    )

    assert var_ind != var_str


def test_risk_engine_loads_sector_map_from_json(monkeypatch, tmp_path):
    from council import risk_engine as risk_mod
    from council.risk_engine import Position, RiskEngine

    sector_map_path = tmp_path / "sector_map.json"
    sector_map_path.write_text('{"AAPL": "Custom Tech"}\n')
    monkeypatch.setattr(risk_mod, "_DEFAULT_SECTOR_MAP_PATH", sector_map_path)

    engine = RiskEngine()
    report = engine.compute_exposure(
        positions=[
            Position(symbol="AAPL", quantity=10, avg_price=100.0, current_price=110.0)
        ],
        portfolio_value=1100.0,
    )

    assert report.sector_exposure == {"Custom Tech": 1100.0}
    assert report.sector_weights == {"Custom Tech": 1.0}


def test_risk_engine_accepts_constructor_sector_map_and_warns_on_unknown_ticker(caplog):
    from council.risk_engine import Position, RiskEngine

    engine = RiskEngine(sector_map={"AAPL": "Custom Tech"})
    positions = [
        Position(symbol="AAPL", quantity=5, avg_price=100.0, current_price=110.0),
        Position(symbol="UNMAPPED", quantity=2, avg_price=50.0, current_price=50.0),
    ]

    with caplog.at_level("WARNING"):
        report = engine.compute_exposure(positions=positions, portfolio_value=650.0)

    assert report.sector_exposure["Custom Tech"] == 550.0
    assert report.sector_exposure["Other"] == 100.0
    assert "Unknown sector mapping for ticker UNMAPPED" in caplog.text


def test_create_positions_from_broker_uses_sector_map_json(monkeypatch, tmp_path):
    from council import risk_engine as risk_mod

    sector_map_path = tmp_path / "sector_map.json"
    sector_map_path.write_text('{"AAPL": "Custom Tech"}\n')
    monkeypatch.setattr(risk_mod, "_DEFAULT_SECTOR_MAP_PATH", sector_map_path)

    positions = risk_mod.create_positions_from_broker(
        pd.DataFrame(
            [
                {
                    "symbol": "AAPL",
                    "qty": 3,
                    "avg_price": 100.0,
                    "current_price": 105.0,
                },
                {
                    "symbol": "UNMAPPED",
                    "qty": 1,
                    "avg_price": 50.0,
                    "current_price": 55.0,
                },
            ]
        )
    )

    assert positions[0].sector == "Custom Tech"
    assert positions[1].sector == "Other"
