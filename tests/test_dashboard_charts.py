import pandas as pd

from dashboard.charts import _annotation_color, regime_timeline, weight_evolution_chart


def test_weight_evolution_chart_uses_valid_fillcolor():
    weights_history = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-04-01", "2026-04-02"]),
            "lgbm": [0.5, 0.45],
            "sentiment": [0.3, 0.35],
            "hmm": [0.2, 0.2],
        }
    )

    fig = weight_evolution_chart(weights_history)

    assert len(fig.data) == 3
    assert [trace.fillcolor for trace in fig.data] == [
        "rgba(31, 119, 180, 0.25)",
        "rgba(255, 127, 14, 0.25)",
        "rgba(44, 160, 44, 0.25)",
    ]


def test_equity_curve_chart_omits_drawdown_overlay_and_handles_gaps():
    from dashboard.charts import equity_curve_chart

    equity = pd.Series(
        [100.0, 102.0, 104.0, 120.0],
        index=pd.to_datetime(["2026-04-01", "2026-04-02", "2026-04-03", "2026-04-14"]),
    )
    benchmark = pd.Series(
        [100.0, 100.5, 101.0, 102.0],
        index=equity.index,
    )

    fig = equity_curve_chart(equity, benchmark)

    assert fig.layout.title.text == "Equity Curve vs SPY (normalized to 100)"
    assert len(fig.data) == 4
    assert {trace.name for trace in fig.data if trace.name is not None} == {"ML Council", "SPY"}
    assert any("Gap detected in equity curve" in ann.text for ann in fig.layout.annotations)


def test_regime_timeline_uses_valid_annotation_colors():
    assert _annotation_color("rgba(44, 160, 44, 0.25)") == "rgb(44, 160, 44)"

    regime_history = pd.DataFrame(
        {
            "date": pd.date_range("2026-04-01", periods=12, freq="D"),
            "regime": ["bull"] * 12,
            "prob_bull": [0.7] * 12,
            "prob_bear": [0.2] * 12,
            "prob_transition": [0.1] * 12,
        }
    )
    equity = pd.Series(
        [100 + i for i in range(12)],
        index=pd.date_range("2026-04-01", periods=12, freq="D"),
    )

    fig = regime_timeline(regime_history, equity)

    assert fig.layout.title.text == "Market Regime History"
    assert len(fig.data) == 4
