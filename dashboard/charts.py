"""Plotly chart builders for the MLCouncil Streamlit dashboard.

All functions return go.Figure instances configured with the dark theme.
They are intentionally pure functions (no st.* calls) so they can be tested
and reused outside Streamlit.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# --------------------------------------------------------------------------
# Shared theme defaults
# --------------------------------------------------------------------------

_DARK_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#0e1117",
    plot_bgcolor="#0e1117",
    font=dict(family="monospace", color="#fafafa"),
    margin=dict(l=60, r=30, t=50, b=50),
)

_MODEL_COLORS = {
    "lgbm":       "#1f77b4",
    "sentiment":  "#ff7f0e",
    "hmm":        "#2ca02c",
}
_REGIME_COLORS = {
    "bull":       "rgba(44, 160, 44, 0.25)",
    "bear":       "rgba(214, 39, 40, 0.25)",
    "transition": "rgba(255, 187, 0, 0.25)",
}


def _color_with_alpha(color: str, alpha: float = 0.25) -> str:
    """Convert a hex color to rgba while preserving existing rgb/rgba colors."""
    if color.startswith("rgba(") or color.startswith("rgb("):
        return color
    if color.startswith("#") and len(color) == 7:
        red = int(color[1:3], 16)
        green = int(color[3:5], 16)
        blue = int(color[5:7], 16)
        return f"rgba({red}, {green}, {blue}, {alpha})"
    return color


def _annotation_color(color: str) -> str:
    """Convert rgba fill colors into rgb annotation colors accepted by Plotly."""
    if color.startswith("rgba(") and color.endswith(")"):
        channels = [part.strip() for part in color[5:-1].split(",")]
        if len(channels) >= 3:
            red, green, blue = channels[:3]
            return f"rgb({red}, {green}, {blue})"
    return color


# ============================================================================
# Performance Tab
# ============================================================================

def equity_curve_chart(
    equity: pd.Series,
    benchmark: pd.Series,
) -> go.Figure:
    """Equity curve vs benchmark normalized to 100, with drawdown band.

    Parameters
    ----------
    equity : pd.Series
        Portfolio equity normalized to 100.
    benchmark : pd.Series
        Benchmark (e.g. SPY) normalized to 100.
    """
    fig = go.Figure()

    # Drawdown band: fill below running peak in red
    if not equity.empty:
        rolling_peak = equity.cummax()
        drawdown_upper = rolling_peak
        drawdown_lower = equity

        # Fill only where drawdown > threshold
        in_dd = equity < rolling_peak
        if in_dd.any():
            fig.add_trace(go.Scatter(
                x=equity.index,
                y=rolling_peak.values,
                fill=None,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter(
                x=equity.index,
                y=equity.values,
                fill="tonexty",
                mode="none",
                fillcolor="rgba(214, 39, 40, 0.15)",
                name="Drawdown",
                hoverinfo="skip",
            ))

    # Benchmark
    if not benchmark.empty:
        bm_aligned = benchmark.reindex(equity.index, method="ffill")
        fig.add_trace(go.Scatter(
            x=bm_aligned.index,
            y=bm_aligned.values,
            mode="lines",
            name="SPY",
            line=dict(color="#636EFA", width=1.5, dash="dot"),
            opacity=0.75,
        ))

    # Portfolio equity
    if not equity.empty:
        fig.add_trace(go.Scatter(
            x=equity.index,
            y=equity.values,
            mode="lines",
            name="ML Council",
            line=dict(color="#00CC96", width=2.5),
        ))

    fig.update_layout(
        **_DARK_LAYOUT,
        title="Equity Curve vs SPY (normalized to 100)",
        xaxis_title="Date",
        yaxis_title="Value (base 100)",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.08),
    )
    return fig


def rolling_sharpe_chart(
    returns: pd.Series,
    window: int = 252,
) -> go.Figure:
    """Annualized rolling Sharpe ratio over a moving window.

    Parameters
    ----------
    returns : pd.Series
        Daily return series.
    window : int
        Rolling window in trading days (default 252 = 1 year).
    """
    fig = go.Figure()

    if returns.empty or len(returns) < window:
        fig.update_layout(**_DARK_LAYOUT, title=f"Rolling Sharpe ({window}d) — insufficient data")
        return fig

    rfr_daily = 0.05 / 252
    excess = returns - rfr_daily
    roll_sharpe = (
        excess.rolling(window).mean()
        / returns.rolling(window).std().replace(0, np.nan)
        * np.sqrt(252)
    ).dropna()

    # Color line by sign
    pos = roll_sharpe.clip(lower=0)
    neg = roll_sharpe.clip(upper=0)

    fig.add_trace(go.Scatter(
        x=roll_sharpe.index, y=roll_sharpe.values,
        mode="lines", name=f"Sharpe {window}d",
        line=dict(color="#FFA15A", width=2),
    ))
    fig.add_hline(y=1.0, line_dash="dash", line_color="#00CC96",
                  annotation_text="SR = 1", annotation_position="top right")
    fig.add_hline(y=0.0, line_dash="dot", line_color="white", opacity=0.4)

    fig.update_layout(
        **_DARK_LAYOUT,
        title=f"Rolling Sharpe ({window}d annualized)",
        xaxis_title="Date",
        yaxis_title="Sharpe Ratio",
        hovermode="x unified",
    )
    return fig


def monthly_returns_heatmap(returns: pd.Series) -> go.Figure:
    """Year × month heatmap of monthly returns.

    Parameters
    ----------
    returns : pd.Series
        Daily return series with DatetimeIndex.
    """
    if returns.empty:
        fig = go.Figure()
        fig.update_layout(**_DARK_LAYOUT, title="Monthly Returns — no data")
        return fig

    monthly = returns.resample("ME").apply(lambda x: (1 + x).prod() - 1) * 100
    table = monthly.groupby([monthly.index.year, monthly.index.month]).sum().unstack(level=1)
    month_labels = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]
    table.columns = [month_labels[m - 1] for m in table.columns]
    year_labels = [str(y) for y in table.index]
    z = table.values

    # Annotation text
    text = [[f"{v:.1f}%" if not np.isnan(v) else "" for v in row] for row in z]

    fig = go.Figure(go.Heatmap(
        z=z,
        x=table.columns.tolist(),
        y=year_labels,
        colorscale=[
            [0.0,  "#d62728"],
            [0.45, "#8B0000"],
            [0.5,  "#262730"],
            [0.55, "#006400"],
            [1.0,  "#2ca02c"],
        ],
        zmid=0,
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=11, family="monospace"),
        showscale=True,
        colorbar=dict(title="Return %", ticksuffix="%"),
    ))
    fig.update_layout(
        **_DARK_LAYOUT,
        title="Monthly Returns Heatmap (%)",
        xaxis_title="Month",
        yaxis_title="Year",
        yaxis=dict(autorange="reversed"),
    )
    return fig


def drawdown_chart(equity: pd.Series) -> go.Figure:
    """Underwater drawdown chart.

    Parameters
    ----------
    equity : pd.Series
        Portfolio equity series.
    """
    if equity.empty:
        fig = go.Figure()
        fig.update_layout(**_DARK_LAYOUT, title="Drawdown — no data")
        return fig

    rolling_max = equity.cummax()
    dd = (equity - rolling_max) / rolling_max * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dd.index,
        y=dd.values,
        fill="tozeroy",
        mode="lines",
        name="Drawdown",
        line=dict(color="#EF553B", width=1.5),
        fillcolor="rgba(239, 85, 59, 0.25)",
    ))
    # Max DD annotation
    max_dd_val = float(dd.min())
    max_dd_date = dd.idxmin()
    fig.add_annotation(
        x=max_dd_date,
        y=max_dd_val,
        text=f"Max DD: {max_dd_val:.1f}%",
        arrowhead=2,
        arrowcolor="#EF553B",
        font=dict(color="#EF553B", size=11),
        bgcolor="#0e1117",
    )
    fig.update_layout(
        **_DARK_LAYOUT,
        title="Drawdown (Underwater Chart)",
        xaxis_title="Date",
        yaxis_title="Drawdown %",
        hovermode="x unified",
    )
    return fig


# ============================================================================
# Council Attribution Tab
# ============================================================================

def model_contribution_bar(
    attribution: pd.DataFrame,
    target_date: Optional[date] = None,
) -> go.Figure:
    """Horizontal bar chart of P&L contribution per council model.

    Parameters
    ----------
    attribution : pd.DataFrame
        Columns: model_name, pnl_contribution, weight, ic_rolling_30d, ...
    target_date : date, optional
        Filter to a specific date (default: latest available).
    """
    fig = go.Figure()

    if attribution.empty:
        fig.update_layout(**_DARK_LAYOUT, title="Model Contribution — no data")
        return fig

    df = attribution.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        if target_date is not None:
            df = df[df["date"].dt.date == target_date]
        else:
            df = df[df["date"] == df["date"].max()]

    if df.empty:
        fig.update_layout(**_DARK_LAYOUT, title="Model Contribution — no data for selected date")
        return fig

    df = df.sort_values("pnl_contribution", ascending=True)

    colors = [
        "#2ca02c" if v >= 0 else "#d62728"
        for v in df["pnl_contribution"]
    ]
    labels = [
        f"{v:+.4f}" for v in df["pnl_contribution"]
    ]

    fig.add_trace(go.Bar(
        x=df["pnl_contribution"],
        y=df["model_name"].str.upper(),
        orientation="h",
        marker_color=colors,
        text=labels,
        textposition="outside",
        textfont=dict(size=12, family="monospace"),
        customdata=np.stack([
            df["weight"].values,
            df["ic_rolling_30d"].values,
        ], axis=-1),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "P&L Contribution: %{x:.4f}<br>"
            "Weight: %{customdata[0]:.1%}<br>"
            "IC (30d): %{customdata[1]:.4f}<extra></extra>"
        ),
    ))

    fig.update_layout(
        **_DARK_LAYOUT,
        title=f"Council P&L Attribution — {target_date or 'Latest'}",
        xaxis_title="P&L Contribution (IC × Weight)",
        yaxis_title="Model",
        showlegend=False,
    )
    fig.add_vline(x=0, line_dash="dash", line_color="white", opacity=0.5)
    return fig


def ic_rolling_chart(ic_history: pd.DataFrame) -> go.Figure:
    """Rolling 30-day IC per model with alert line at IC = 0.01.

    Parameters
    ----------
    ic_history : pd.DataFrame
        Columns: date, lgbm, sentiment, hmm  (one column per model).
    """
    fig = go.Figure()

    if ic_history.empty:
        fig.update_layout(**_DARK_LAYOUT, title="IC Rolling 30d — no data")
        return fig

    model_cols = [c for c in ic_history.columns if c != "date"]
    for model in model_cols:
        if model not in ic_history.columns:
            continue
        color = _MODEL_COLORS.get(model, "#888888")
        fig.add_trace(go.Scatter(
            x=ic_history["date"],
            y=ic_history[model],
            mode="lines",
            name=model.upper(),
            line=dict(color=color, width=2),
        ))

    # Alert: IC = 0.01 threshold
    fig.add_hline(
        y=0.01,
        line_dash="dash",
        line_color="#d62728",
        annotation_text="IC alert (0.01)",
        annotation_position="top right",
        annotation_font=dict(color="#d62728", size=10),
    )
    fig.add_hline(y=0.0, line_dash="dot", line_color="white", opacity=0.3)

    fig.update_layout(
        **_DARK_LAYOUT,
        title="IC Rolling 30d per Model",
        xaxis_title="Date",
        yaxis_title="Information Coefficient",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.08),
    )
    return fig


def weight_evolution_chart(weights_history: pd.DataFrame) -> go.Figure:
    """Stacked area chart of council model weights over time.

    Parameters
    ----------
    weights_history : pd.DataFrame
        Columns: date, lgbm, sentiment, hmm  (weights, sum ~1 per row).
    """
    fig = go.Figure()

    if weights_history.empty:
        fig.update_layout(**_DARK_LAYOUT, title="Weight Evolution — no data")
        return fig

    model_cols = [c for c in weights_history.columns if c != "date"]

    for i, model in enumerate(model_cols):
        if model not in weights_history.columns:
            continue
        color = _MODEL_COLORS.get(model, f"hsl({i * 120}, 70%, 50%)")
        fig.add_trace(go.Scatter(
            x=weights_history["date"],
            y=weights_history[model],
            mode="lines",
            name=model.upper(),
            stackgroup="one",
            line=dict(width=0.5, color=color),
            fillcolor=_color_with_alpha(color),
        ))

    fig.update_layout(
        **_DARK_LAYOUT,
        title="Council Weight Evolution",
        xaxis_title="Date",
        yaxis_title="Weight",
        yaxis=dict(tickformat=".0%", range=[0, 1]),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.08),
    )
    return fig


# ============================================================================
# Regime Tab
# ============================================================================

def regime_timeline(
    regime_history: pd.DataFrame,
    equity: Optional[pd.Series] = None,
) -> go.Figure:
    """Timeline of regime bands with equity curve overlay.

    Parameters
    ----------
    regime_history : pd.DataFrame
        Columns: date, regime (str), prob_bull, prob_bear, prob_transition.
    equity : pd.Series, optional
        Normalized equity curve to overlay on secondary axis.
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    if regime_history.empty:
        fig.update_layout(**_DARK_LAYOUT, title="Regime Timeline — no data")
        return fig

    df = regime_history.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    # Draw colored regime bands using vrect
    i = 0
    while i < len(df):
        regime = df.iloc[i]["regime"]
        start = df.iloc[i]["date"]
        # Find end of this regime block
        j = i + 1
        while j < len(df) and df.iloc[j]["regime"] == regime:
            j += 1
        end = df.iloc[j]["date"] if j < len(df) else df.iloc[-1]["date"]
        color = _REGIME_COLORS.get(regime, "rgba(128,128,128,0.2)")

        fig.add_vrect(
            x0=start, x1=end,
            fillcolor=color,
            layer="below",
            line_width=0,
            annotation_text=regime.capitalize() if (j - i) > 10 else "",
            annotation_position="top left",
            annotation_font=dict(size=9, color=_annotation_color(color)),
        )
        i = j

    # Equity overlay on secondary y-axis
    if equity is not None and not equity.empty:
        fig.add_trace(
            go.Scatter(
                x=equity.index,
                y=equity.values,
                mode="lines",
                name="Equity",
                line=dict(color="#00CC96", width=2),
            ),
            secondary_y=True,
        )

    # Regime label trace for legend
    for regime, color in [("Bull", "#2ca02c"), ("Bear", "#d62728"), ("Transition", "#ffbb00")]:
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(size=10, color=color, symbol="square"),
            name=regime,
            showlegend=True,
        ))

    fig.update_layout(
        **_DARK_LAYOUT,
        title="Market Regime History",
        xaxis_title="Date",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.08),
    )
    fig.update_yaxes(title_text="Equity (base 100)", secondary_y=True)
    fig.update_yaxes(visible=False, secondary_y=False)
    return fig


def regime_probability_gauge(probs: dict) -> go.Figure:
    """Three gauge charts for bull, bear, and transition probabilities.

    Parameters
    ----------
    probs : dict
        Keys: bull, bear, transition  (float 0–1 probabilities).
    """
    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}]],
        subplot_titles=["Bull", "Transition", "Bear"],
    )

    gauge_cfgs = [
        ("bull",       probs.get("bull", 0),       "#2ca02c", 1, 1),
        ("transition", probs.get("transition", 0), "#ffbb00", 1, 2),
        ("bear",       probs.get("bear", 0),        "#d62728", 1, 3),
    ]

    for name, value, color, row, col in gauge_cfgs:
        pct = value * 100
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=pct,
                number=dict(suffix="%", font=dict(size=24, color=color)),
                gauge=dict(
                    axis=dict(range=[0, 100], tickcolor="#fafafa"),
                    bar=dict(color=color),
                    bgcolor="#262730",
                    bordercolor="#fafafa",
                    steps=[
                        dict(range=[0, 33], color="#1a1a2e"),
                        dict(range=[33, 66], color="#16213e"),
                        dict(range=[66, 100], color="#0f3460"),
                    ],
                    threshold=dict(
                        line=dict(color=color, width=3),
                        thickness=0.75,
                        value=pct,
                    ),
                ),
            ),
            row=row, col=col,
        )

    fig.update_layout(
        **_DARK_LAYOUT,
        title="Current Regime Probabilities",
        height=280,
    )
    return fig


def current_weights_radar(weights: dict) -> go.Figure:
    """Radar chart of current council weights per model.

    Parameters
    ----------
    weights : dict
        model_name → weight (float).  E.g. {"lgbm": 0.5, "sentiment": 0.3, "hmm": 0.2}
    """
    if not weights:
        fig = go.Figure()
        fig.update_layout(**_DARK_LAYOUT, title="Current Weights — no data")
        return fig

    categories = [k.upper() for k in weights]
    values = list(weights.values())
    # Close the loop
    categories_closed = categories + [categories[0]]
    values_closed = values + [values[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill="toself",
        fillcolor="rgba(31, 119, 180, 0.3)",
        line=dict(color="#1f77b4", width=2),
        name="Current Weights",
        mode="lines+markers",
        marker=dict(size=8, color="#1f77b4"),
    ))

    fig.update_layout(
        **_DARK_LAYOUT,
        title="Council Weights (Current Regime)",
        polar=dict(
            bgcolor="#262730",
            radialaxis=dict(
                visible=True,
                range=[0, max(max(values) * 1.2, 0.1)],
                tickformat=".0%",
                color="#fafafa",
                gridcolor="#444",
            ),
            angularaxis=dict(color="#fafafa", gridcolor="#444"),
        ),
        showlegend=False,
        height=380,
    )
    return fig
