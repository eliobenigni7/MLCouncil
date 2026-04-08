import pandas as pd

from dashboard.charts import weight_evolution_chart


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
