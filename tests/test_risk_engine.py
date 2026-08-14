from __future__ import annotations
from datetime import datetime, timezone

import numpy as np
import pandas as pd


def test_risk_engine_flags_position_and_sector_breaches():
    from council.risk.risk_engine import (
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
    from council.risk import risk_engine as risk_mod
    from council.risk.risk_engine import Position, RiskEngine

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
    from council.risk.risk_engine import Position, RiskEngine

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
    from council.risk.risk_engine import Position, RiskEngine

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
    from council.risk.risk_engine import RiskEngine

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
    from council.risk import risk_engine as risk_mod
    from council.risk.risk_engine import Position, RiskEngine

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
    from council.risk.risk_engine import Position, RiskEngine

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
    from council.risk import risk_engine as risk_mod

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


# ---------------------------------------------------------------------------
# M10 multivariate Monte Carlo VaR upgrade (docs/math-drilldown section 1)
# ---------------------------------------------------------------------------


def _kupiec_pof_lr(n_exceed: int, n_total: int, p: float) -> float:
    """Kupiec POF (unconditional coverage) likelihood ratio, ~ chi2_1 under H0.

    LR_POF = -2 ln[ ((1-p)^(T-N) p^N) / ((1-N/T)^(T-N) (N/T)^N) ]
    """
    if n_exceed in (0, n_total):
        return float("inf")
    pi_hat = n_exceed / n_total
    num = (1.0 - p) ** (n_total - n_exceed) * p ** n_exceed
    den = (1.0 - pi_hat) ** (n_total - n_exceed) * pi_hat ** n_exceed
    return -2.0 * np.log(num / den) if num > 0.0 and den > 0.0 else float("inf")


def _christoffersen_ind_lr(violations: np.ndarray) -> float:
    """Christoffersen (independence) likelihood ratio, ~ chi2_1 under H0.

    LR_ind = -2 ln[ (1-pi)^(n00+n10) pi^(n01+n11)
                    / ((1-pi0)^n00 pi0^n01 (1-pi1)^n10 pi1^n11) ]
    Uses the 0 * log(0) = 0 convention for empty transition cells.
    """
    v = np.asarray(violations, dtype=bool)
    n_exceed = int(v.sum())
    if n_exceed in (0, len(v)):
        return 0.0
    n01 = int((~v[:-1] & v[1:]).sum())
    n00 = int((~v[:-1] & ~v[1:]).sum())
    n10 = int((v[:-1] & ~v[1:]).sum())
    n11 = int((v[:-1] & v[1:]).sum())

    def _clog(count: int, prob: float) -> float:
        return 0.0 if count == 0 else count * np.log(prob)

    pi0 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi1 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    return float(
        -2.0
        * (
            _clog(n00 + n10, 1.0 - pi)
            + _clog(n01 + n11, pi)
            - _clog(n00, 1.0 - pi0)
            - _clog(n01, pi0)
            - _clog(n10, 1.0 - pi1)
            - _clog(n11, pi1)
        )
    )


def test_kupiec_pof_and_christoffersen_backtest():
    """Backtest statistics behave: correct model passes, wrong model fails.

    On synthetic data with known parameters, a correct VaR model keeps both
    likelihood ratios below the chi2_1 95% critical value (3.84), while an
    understated vol (LR_POF) and clustered violations (LR_ind) reject.
    """
    rng = np.random.default_rng(42)
    n_total = 2000
    sigma = 0.01
    p = 0.01
    z_99 = 2.326  # Phi^-1(0.99)

    # Correct model: VaR built from the true volatility.
    returns = rng.normal(0.0, sigma, size=n_total)
    violations = returns < -z_99 * sigma
    assert _kupiec_pof_lr(int(violations.sum()), n_total, p) < 3.84
    assert _christoffersen_ind_lr(violations) < 3.84

    # Wrong model: half the true vol understates VaR -> mass violations.
    violations = returns < -z_99 * (sigma / 2.0)
    assert _kupiec_pof_lr(int(violations.sum()), n_total, p) > 3.84

    # Vol-clustered data with a model calibrated on the full sample:
    # violations pile up inside the high-vol block -> dependence rejected.
    clustered = rng.normal(0.0, sigma, size=n_total)
    clustered[800:1200] = rng.normal(0.0, 3.0 * sigma, size=400)
    violations = clustered < -z_99 * clustered.std()
    assert _christoffersen_ind_lr(violations) > 3.84
    assert _kupiec_pof_lr(int(violations.sum()), n_total, p) > 3.84


def test_monte_carlo_es_matches_gaussian_closed_form():
    """MC ES on Gaussian data approximates the closed-form Gaussian ES."""
    from scipy.stats import norm as _scipy_norm

    from council.risk.risk_engine import RiskEngine

    rng = np.random.default_rng(11)
    sigma, rho = 0.02, 0.3
    cov_true = np.array(
        [[sigma**2, rho * sigma**2], [rho * sigma**2, sigma**2]]
    )
    returns = pd.DataFrame(
        rng.multivariate_normal([0.0, 0.0], cov_true, size=2000),
        columns=["AAA", "BBB"],
    )
    weights = {"AAA": 0.6, "BBB": 0.4}
    portfolio_value = 1_000_000.0
    w = np.array([0.6, 0.4])
    sigma_p = np.sqrt(float(w @ cov_true @ w))
    z = _scipy_norm.ppf(0.99)
    var_closed = z * sigma_p * portfolio_value
    es_closed = sigma_p * _scipy_norm.pdf(z) / 0.01 * portfolio_value

    result = RiskEngine(seed=3).compute_var_monte_carlo(
        returns,
        weights,
        portfolio_value,
        n_simulations=100_000,
        confidence=0.99,
        horizon=1,
        tail_dof=None,
    )

    # ES must be reported alongside VaR and dominate it.
    assert result.es_dollar > result.var_dollar
    assert np.isclose(result.var_dollar, var_closed, rtol=0.05)
    assert np.isclose(result.es_dollar, es_closed, rtol=0.05)
    # Legacy 2-tuple unpacking keeps working.
    var_unpacked, cvar_unpacked = result
    assert var_unpacked == result.var_dollar
    assert cvar_unpacked == result.cvar_dollar


def test_t_copula_tail_dependence_exceeds_gaussian():
    """t-copula shows lower-tail dependence; the Gaussian copula does not.

    Empirical co-exceedance rate lambda_L = P(X <= q_p | Y <= q_p) at a deep
    threshold (p = 0.0005): t(nu=5, rho=0.5) must exceed 0.1, Gaussian must
    stay near 0 (lambda_L = 0 asymptotically for the Gaussian copula).
    """
    from council.risk.risk_engine import RiskEngine

    n_obs = 2000
    rng = np.random.default_rng(5)
    cov_target = np.array([[1.0, 0.5], [0.5, 1.0]]) * (0.02**2)
    # Deterministic sample covariance == cov_target (QR construction).
    base = rng.normal(size=(n_obs, 2))
    base = base - base.mean(axis=0, keepdims=True)
    q, _ = np.linalg.qr(base)
    x = np.sqrt(n_obs - 1.0) * q[:, :2] @ np.linalg.cholesky(cov_target).T
    returns = pd.DataFrame(x, columns=["AAA", "BBB"])

    engine = RiskEngine(seed=1)
    draws_t = engine.simulate_daily_returns(
        returns, n_simulations=2_000_000, horizon=1, tail_dof=5, seed=7
    )[:, 0, :]
    draws_g = engine.simulate_daily_returns(
        returns, n_simulations=2_000_000, horizon=1, tail_dof=None, seed=7
    )[:, 0, :]

    def empirical_lambda_l(draws: np.ndarray, p: float) -> float:
        q_x = np.quantile(draws[:, 0], p)
        q_y = np.quantile(draws[:, 1], p)
        in_y_tail = draws[:, 1] <= q_y
        if not in_y_tail.any():
            return 0.0
        return float((draws[in_y_tail, 0] <= q_x).mean())

    lam_t = empirical_lambda_l(draws_t, 0.0005)
    lam_g = empirical_lambda_l(draws_g, 0.0005)
    assert lam_t > 0.10
    assert lam_g < 0.05


def test_monte_carlo_multi_step_10d_var_compounding():
    """Multi-step compounding: 10d VaR > 1d VaR and differs from sqrt(10).

    On vol-clustered data (elevated regime at the end of the sample) the
    GARCH(1,1) daily covariance path makes the compounded 10-day VaR deviate
    from the naive sqrt(10) * 1d VaR scaling.
    """
    from council.risk.risk_engine import RiskEngine

    rng = np.random.default_rng(9)
    n_days = 600
    vols = np.full(n_days, 0.01)
    vols[-150:] = 0.03  # recent vol cluster
    factor = rng.normal(0.0, 1.0, size=n_days)
    e1 = rng.normal(0.0, 1.0, size=n_days)
    e2 = rng.normal(0.0, 1.0, size=n_days)
    returns = pd.DataFrame(
        {
            "AAA": vols * (0.7 * factor + 0.3 * e1),
            "BBB": vols * (0.7 * factor + 0.3 * e2),
        }
    )
    weights = {"AAA": 0.5, "BBB": 0.5}
    portfolio_value = 1_000_000.0

    engine = RiskEngine(seed=13)
    var_1d = engine.compute_var_monte_carlo(
        returns,
        weights,
        portfolio_value,
        n_simulations=50_000,
        confidence=0.99,
        horizon=1,
        seed=11,
        tail_dof=None,
    )
    var_10d = engine.compute_var_monte_carlo(
        returns,
        weights,
        portfolio_value,
        n_simulations=50_000,
        confidence=0.99,
        horizon=10,
        seed=11,
        tail_dof=None,
    )

    # Monotonicity in the horizon.
    assert var_10d.var_dollar > var_1d.var_dollar
    # Compounded 10d VaR deviates from sqrt(10) * 1d VaR under vol clustering.
    assert not np.isclose(
        var_10d.var_dollar, np.sqrt(10.0) * var_1d.var_dollar, rtol=0.08
    )


def test_monte_carlo_stress_replay_increases_var():
    """Stress replay (correlation 0.9 + eigenvalue shock) raises the MC VaR."""
    from council.risk.risk_engine import RiskEngine

    rng = np.random.default_rng(9)
    n_days = 600
    vols = np.full(n_days, 0.01)
    vols[-150:] = 0.03
    factor = rng.normal(0.0, 1.0, size=n_days)
    e1 = rng.normal(0.0, 1.0, size=n_days)
    e2 = rng.normal(0.0, 1.0, size=n_days)
    returns = pd.DataFrame(
        {
            "AAA": vols * (0.7 * factor + 0.3 * e1),
            "BBB": vols * (0.7 * factor + 0.3 * e2),
        }
    )
    weights = {"AAA": 0.5, "BBB": 0.5}
    portfolio_value = 1_000_000.0

    engine = RiskEngine(seed=13)
    base = engine.compute_var_monte_carlo(
        returns,
        weights,
        portfolio_value,
        n_simulations=50_000,
        confidence=0.99,
        horizon=1,
        seed=11,
        tail_dof=None,
    )
    stressed = engine.compute_var_monte_carlo(
        returns,
        weights,
        portfolio_value,
        n_simulations=50_000,
        confidence=0.99,
        horizon=1,
        seed=11,
        tail_dof=None,
        stress_replay=True,
    )

    assert stressed.var_dollar > base.var_dollar
