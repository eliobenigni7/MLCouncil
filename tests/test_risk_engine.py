from __future__ import annotations

from datetime import datetime, timezone

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
