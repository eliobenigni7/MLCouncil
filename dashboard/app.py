"""MLCouncil — Streamlit Trading Dashboard

Three-tab dashboard:
  1. Performance  — equity curve, rolling Sharpe, monthly heatmap, drawdown
  2. Council Attribution — model contributions, IC rolling, weight evolution
  3. Regime       — regime timeline, probability gauges, radar weights

Deploy free on Streamlit Cloud:
  streamlit run dashboard/app.py

Public-safe: only normalized metrics are displayed (no raw capital/orders).
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import streamlit as st

# Ensure project root is importable
_ROOT = Path(__file__).parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import dashboard.charts as charts
from council.alerts import load_current_alerts
from dashboard.data_loader import (
    load_benchmark,
    load_daily_returns,
    load_equity_curve,
    load_ic_history,
    load_model_attribution,
    load_regime_history,
    load_current_regime,
    load_sidebar_metrics,
    load_weights_history,
)

# ============================================================================
# Page config — must be first Streamlit call
# ============================================================================

st.set_page_config(
    page_title="ML Council — Trading Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# Helper: live sidebar metrics
# ============================================================================

def render_live_metrics() -> None:
    """Three-metric row updated via 5-min TTL cache."""
    metrics = load_sidebar_metrics()

    regime = metrics.get("regime", "N/A")
    regime_emoji = {"Bull": "🟢", "Bear": "🔴", "Transition": "🟡"}.get(regime, "⚪")
    regime_prob = metrics.get("regime_prob", 0.0)
    st.metric(
        "Current Regime",
        value=f"{regime_emoji} {regime}",
        delta=f"prob: {regime_prob:.0f}%",
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        sharpe_ytd = metrics.get("sharpe_ytd", 0.0)
        sharpe_delta = metrics.get("sharpe_delta", 0.0)
        st.metric("Sharpe YTD", f"{sharpe_ytd:.2f}", delta=f"{sharpe_delta:+.3f}")
    with col2:
        max_dd = metrics.get("max_dd", 0.0)
        st.metric("Max DD", f"{max_dd:.1f}%", delta_color="inverse")
    with col3:
        ic_30d = metrics.get("ic_30d", 0.0)
        ic_delta = metrics.get("ic_delta", 0.0)
        st.metric("IC 30d", f"{ic_30d:.4f}", delta=f"{ic_delta:+.4f}")


# ============================================================================
# Tab: Performance
# ============================================================================

def render_performance_tab(mode: str, start_date: date, end_date: date) -> None:
    """Equity curve, rolling Sharpe, monthly heatmap, drawdown."""
    equity = load_equity_curve(mode)
    benchmark = load_benchmark(mode)
    returns = load_daily_returns(mode)

    # KPI row
    if not returns.empty and len(returns) > 1:
        rfr = 0.05 / 252
        sharpe = float(
            (returns - rfr).mean() / returns.std() * (252 ** 0.5)
        ) if returns.std() > 0 else 0.0
        rolling_max = equity.cummax()
        max_dd = float(((equity - rolling_max) / rolling_max).min() * 100)
        cagr = float(
            ((equity.iloc[-1] / equity.iloc[0]) ** (252 / max(len(equity), 1)) - 1) * 100
        )

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Sharpe (full period)", f"{sharpe:.2f}")
        k2.metric("Max Drawdown", f"{max_dd:.1f}%")
        k3.metric("CAGR (ann.)", f"{cagr:.1f}%")
        k4.metric("Trading Days", str(len(equity)))

    st.divider()

    # Equity curve (full width)
    st.plotly_chart(
        charts.equity_curve_chart(equity, benchmark),
        use_container_width=True,
        key=f"performance_equity_{mode}",
    )

    # Rolling Sharpe + Drawdown
    col_l, col_r = st.columns(2)
    with col_l:
        window = st.select_slider(
            "Rolling Sharpe window (days)",
            options=[63, 126, 252],
            value=252,
            key="sharpe_window",
        )
        st.plotly_chart(
            charts.rolling_sharpe_chart(returns, window=window),
            use_container_width=True,
            key=f"performance_rolling_sharpe_{mode}_{window}",
        )
    with col_r:
        st.plotly_chart(
            charts.drawdown_chart(equity),
            use_container_width=True,
            key=f"performance_drawdown_{mode}",
        )

    # Monthly heatmap
    st.plotly_chart(
        charts.monthly_returns_heatmap(returns),
        use_container_width=True,
        key=f"performance_monthly_heatmap_{mode}",
    )


# ============================================================================
# Tab: Council Attribution
# ============================================================================

def render_attribution_tab(start_date: date, end_date: date) -> None:
    """Per-model P&L contribution, IC rolling, weight evolution."""
    attribution = load_model_attribution(start=start_date, end=end_date)
    ic_history = load_ic_history()
    weights_history = load_weights_history()

    # Date picker
    if not attribution.empty and "date" in attribution.columns:
        available_dates = sorted(attribution["date"].dt.date.unique())
        selected_date = st.select_slider(
            "Attribution date",
            options=available_dates,
            value=available_dates[-1],
            key="attr_date",
        )
    else:
        selected_date = None

    # Contribution bar + weights radar
    col_l, col_r = st.columns([3, 2])
    with col_l:
        st.plotly_chart(
            charts.model_contribution_bar(attribution, selected_date),
            use_container_width=True,
            key=f"attribution_contribution_{selected_date or 'latest'}",
        )
    with col_r:
        if not attribution.empty and selected_date is not None:
            latest = attribution[attribution["date"].dt.date == selected_date]
        elif not attribution.empty:
            latest = attribution[attribution["date"] == attribution["date"].max()]
        else:
            latest = attribution

        weights_dict = (
            dict(zip(latest["model_name"], latest["weight"]))
            if not latest.empty else {}
        )
        st.plotly_chart(
            charts.current_weights_radar(weights_dict),
            use_container_width=True,
            key=f"attribution_weights_radar_{selected_date or 'latest'}",
        )

    # IC rolling
    st.plotly_chart(
        charts.ic_rolling_chart(ic_history),
        use_container_width=True,
        key="attribution_ic_rolling",
    )

    # Weight evolution
    st.plotly_chart(
        charts.weight_evolution_chart(weights_history),
        use_container_width=True,
        key="attribution_weight_evolution",
    )

    # Raw table
    with st.expander("Attribution data table"):
        if not attribution.empty:
            display = attribution.copy()
            if "date" in display.columns:
                display["date"] = display["date"].dt.date
            for col in ["weight", "ic_rolling_30d", "pnl_contribution"]:
                if col in display.columns:
                    display[col] = display[col].round(4)
            st.dataframe(display, use_container_width=True)
        else:
            st.info("No attribution data available.")


# ============================================================================
# Tab: Regime
# ============================================================================

def render_regime_tab(mode: str, start_date: date, end_date: date) -> None:
    """Regime timeline, probability gauges, current-regime radar."""
    regime_info = load_current_regime()
    regime_history = load_regime_history()
    equity = load_equity_curve(mode)
    attribution = load_model_attribution(start=start_date, end=end_date)

    regime_name = regime_info.get("regime", "unknown").capitalize()
    regime_emoji = {"Bull": "🟢", "Bear": "🔴", "Transition": "🟡"}.get(regime_name, "⚪")
    st.subheader(f"Current Regime: {regime_emoji} {regime_name}")

    # Probability gauges
    st.plotly_chart(
        charts.regime_probability_gauge(regime_info),
        use_container_width=True,
        key=f"regime_probability_{regime_info.get('regime', 'unknown')}",
    )

    # Timeline + radar
    col_l, col_r = st.columns([2, 1])
    with col_l:
        st.plotly_chart(
            charts.regime_timeline(regime_history, equity),
            use_container_width=True,
            key=f"regime_timeline_{mode}",
        )
    with col_r:
        if not attribution.empty:
            latest = attribution[attribution["date"] == attribution["date"].max()]
            weights_dict = dict(zip(latest["model_name"], latest["weight"]))
        else:
            weights_dict = {}

        st.plotly_chart(
            charts.current_weights_radar(weights_dict),
            use_container_width=True,
            key=f"regime_weights_radar_{regime_info.get('regime', 'unknown')}",
        )

    # Regime stats
    if not regime_history.empty:
        with st.expander("Regime statistics"):
            stats = (
                regime_history.groupby("regime")
                .agg(
                    days=("date", "count"),
                    avg_bull_prob=("prob_bull", "mean"),
                    avg_bear_prob=("prob_bear", "mean"),
                    avg_transition_prob=("prob_transition", "mean"),
                )
                .reset_index()
            )
            for col in ["avg_bull_prob", "avg_bear_prob", "avg_transition_prob"]:
                stats[col] = stats[col].round(3)
            st.dataframe(stats, use_container_width=True)


# ============================================================================
# Sidebar: monitoring alerts
# ============================================================================

def _render_sidebar_alerts() -> None:
    """Display active monitoring alerts in the sidebar.

    Reads from data/monitoring/current_alerts.json (written by AlertDispatcher).
    Shows nothing when there are no active alerts.
    """
    alerts = load_current_alerts()
    active = [a for a in alerts if a.get("is_alert", False)]
    if not active:
        st.caption("No active alerts.")
        return

    st.subheader("Monitoring Alerts")
    for alert in active:
        severity = alert.get("severity", "info")
        model = alert.get("model_name", "")
        check = alert.get("check_type", "")
        message = alert.get("message", "")
        recommendation = alert.get("recommendation", "")

        label = f"**[{check.upper()}]** {model}: {message}"
        detail = f"Recommendation: {recommendation}" if recommendation else ""

        if severity == "critical":
            st.error(f"🚨 {label}")
            if detail:
                st.caption(detail)
        elif severity == "warning":
            st.warning(f"⚠️ {label}")
            if detail:
                st.caption(detail)
        else:
            st.info(f"ℹ️ {label}")


# ============================================================================
# Sidebar
# ============================================================================

with st.sidebar:
    st.title("ML Council")
    st.caption("Multi-model ensemble trading")

    mode = st.radio("Mode", ["Paper Trading", "Backtest"], horizontal=True)

    if mode == "Backtest":
        col_s, col_e = st.columns(2)
        with col_s:
            start_date = st.date_input("Start", value=date.today() - timedelta(days=730))
        with col_e:
            end_date = st.date_input("End", value=date.today())
    else:
        start_date = date.today() - timedelta(days=180)
        end_date = date.today()

    st.divider()
    render_live_metrics()

    st.divider()
    _render_sidebar_alerts()

    st.divider()
    st.caption("Equity/attribution: 1h cache · Regime: 5 min cache")
    if st.button("Force refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


# ============================================================================
# Main tabs
# ============================================================================

tab1, tab2, tab3 = st.tabs(["📈 Performance", "🧠 Council Attribution", "🔄 Regime"])

with tab1:
    render_performance_tab(mode, start_date, end_date)

with tab2:
    render_attribution_tab(start_date, end_date)

with tab3:
    render_regime_tab(mode, start_date, end_date)
