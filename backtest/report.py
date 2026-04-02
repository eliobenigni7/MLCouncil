"""BacktestReport — metriche quantitative e report HTML interattivo.

Metriche implementate
---------------------
- Sharpe ratio (annualizzato, risk-free 5%)
- Max drawdown (in %)
- Calmar ratio (CAGR / |max_dd|)
- Information Coefficient (IC medio e rolling 30d per modello)
- Turnover giornaliero (% del portafoglio)
- Model attribution (P&L contribuzione per modello council)

Report HTML
-----------
Genera un file HTML con grafici Plotly interattivi:
  1. Equity curve vs SPY benchmark
  2. Drawdown chart
  3. Rolling Sharpe (252 giorni)
  4. IC per modello (rolling 30d)
  5. Heatmap rendimenti mensili

Utilizzo
--------
    from backtest.report import BacktestReport

    report = BacktestReport(fills=result.fills, prices=prices_df)
    print(f"Sharpe: {report.sharpe_ratio():.3f}")
    print(f"Max DD: {report.max_drawdown():.1%}")
    report.generate_html_report("output/report.html")
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

_ROOT = Path(__file__).parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ===========================================================================
# BacktestReport
# ===========================================================================

class BacktestReport:
    """Report quantitativo completo per un backtest MLCouncil.

    Parameters
    ----------
    fills : pd.DataFrame
        Fill report da NautilusTrader o fill sintetici.
        Colonne attese: ts_event (ns), instrument_id, last_px, last_qty,
        order_side, realized_pnl (opzionale).
    prices : pd.DataFrame
        OHLCV wide (index=date, columns=ticker, values=adj_close).
        Usato per benchmark SPY e calcolo rendimenti.
    initial_capital : float
        Capitale iniziale in USD.
    risk_free_rate : float
        Tasso risk-free annuale (default 5%).
    equity_curve : pd.Series, optional
        Se fornita, viene usata direttamente (override del calcolo dai fill).
    """

    def __init__(
        self,
        fills: pd.DataFrame,
        prices: pd.DataFrame,
        initial_capital: float = 100_000.0,
        risk_free_rate: float = 0.05,
        equity_curve: Optional[pd.Series] = None,
    ) -> None:
        self.fills = fills.copy() if not fills.empty else pd.DataFrame()
        self.prices = prices.copy() if not prices.empty else pd.DataFrame()
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate

        # Equity curve: usa quella fornita o ricostruisce dai fill
        if equity_curve is not None and not equity_curve.empty:
            self.equity_curve = equity_curve.copy()
        else:
            self.equity_curve = self._compute_equity_curve()

        self.returns: pd.Series = (
            self.equity_curve.pct_change().dropna()
            if len(self.equity_curve) > 1
            else pd.Series(dtype=float)
        )

    # ===========================================================================
    # Core metrics
    # ===========================================================================

    def sharpe_ratio(self, risk_free_rate: Optional[float] = None) -> float:
        """Sharpe ratio annualizzato.

        Parameters
        ----------
        risk_free_rate : float, optional
            Override del tasso risk-free (default: self.risk_free_rate).

        Returns
        -------
        float
        """
        rfr = risk_free_rate if risk_free_rate is not None else self.risk_free_rate
        if self.returns.empty or self.returns.std() < 1e-12:
            return 0.0
        daily_rfr = rfr / 252
        excess = self.returns - daily_rfr
        return float(excess.mean() / self.returns.std() * np.sqrt(252))

    def max_drawdown(self) -> float:
        """Maximum drawdown come numero negativo.

        Returns
        -------
        float
            Valore negativo, es. -0.152 = -15.2% drawdown massimo.
        """
        if self.equity_curve.empty:
            return 0.0
        rolling_max = self.equity_curve.cummax()
        dd = (self.equity_curve - rolling_max) / rolling_max
        return float(dd.min())

    def calmar_ratio(self) -> float:
        """CAGR diviso il valore assoluto del max drawdown.

        Returns
        -------
        float
            Infinito se il drawdown è zero.
        """
        cagr = self._cagr()
        mdd = abs(self.max_drawdown())
        if mdd < 1e-9:
            return float("inf")
        return float(cagr / mdd)

    def information_coefficient(
        self,
        signals: pd.DataFrame,
        forward_returns: Optional[pd.DataFrame] = None,
    ) -> float:
        """IC medio tra segnali (cross-sezionali) e rendimenti forward 1d.

        Parameters
        ----------
        signals : pd.DataFrame
            index=date, columns=ticker, values=signal z-scores.
        forward_returns : pd.DataFrame, optional
            index=date, columns=ticker. Se None, usa self.prices per calcolare
            i rendimenti 1-day forward.

        Returns
        -------
        float
            IC medio (Spearman) sull'intero periodo.
        """
        if signals.empty:
            return float("nan")

        if forward_returns is None:
            if self.prices.empty:
                return float("nan")
            forward_returns = self.prices.pct_change().shift(-1)

        ic_values = []
        for d in signals.index:
            if d not in forward_returns.index:
                continue
            sig = signals.loc[d].dropna()
            ret = forward_returns.loc[d].dropna()
            common = sig.index.intersection(ret.index)
            if len(common) < 3:
                continue
            ic_val, _ = spearmanr(sig[common].values, ret[common].values)
            if not np.isnan(ic_val):
                ic_values.append(ic_val)

        return float(np.mean(ic_values)) if ic_values else float("nan")

    def turnover_analysis(self) -> pd.DataFrame:
        """Turnover giornaliero medio e costi totali stimati.

        Returns
        -------
        pd.DataFrame
            Colonne: date, turnover_pct, trade_count, estimated_cost_usd.
        """
        if self.fills.empty:
            return pd.DataFrame(
                columns=["date", "turnover_pct", "trade_count", "estimated_cost_usd"]
            )

        df = self.fills.copy()

        # Normalizza colonne
        ts_col = next(
            (c for c in ["ts_event", "timestamp", "ts_init"] if c in df.columns), None
        )
        if ts_col is None:
            return pd.DataFrame(
                columns=["date", "turnover_pct", "trade_count", "estimated_cost_usd"]
            )

        df["_date"] = pd.to_datetime(df[ts_col], unit="ns", utc=True).dt.date

        px_col  = next((c for c in ["last_px", "fill_price", "avg_px"] if c in df.columns), None)
        qty_col = next((c for c in ["last_qty", "quantity", "qty"] if c in df.columns), None)

        rows = []
        for d, grp in df.groupby("_date"):
            trade_count = len(grp)
            if px_col and qty_col:
                traded_usd = (
                    grp[px_col].astype(float) * grp[qty_col].astype(float)
                ).sum()
                turnover_pct = traded_usd / self.initial_capital
                # Stima costi: 4 bps totali (3 slip + 1 comm)
                estimated_cost = traded_usd * 0.0004
            else:
                traded_usd = 0.0
                turnover_pct = 0.0
                estimated_cost = 0.0

            rows.append({
                "date":               pd.Timestamp(d),
                "turnover_pct":       float(turnover_pct),
                "trade_count":        int(trade_count),
                "estimated_cost_usd": float(estimated_cost),
            })

        if not rows:
            return pd.DataFrame(
                columns=["date", "turnover_pct", "trade_count", "estimated_cost_usd"]
            )
        return pd.DataFrame(rows).set_index("date")

    def model_attribution(
        self,
        model_signals: dict[str, pd.DataFrame],
        forward_returns: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """P&L attribuito a ogni modello del council.

        Calcola il contributo di ciascun modello come:
            IC × peso_medio × vol_portafoglio

        Parameters
        ----------
        model_signals : dict[str, pd.DataFrame]
            model_name → DataFrame(index=date, columns=ticker).
        forward_returns : pd.DataFrame, optional
            index=date, columns=ticker. Se None, usa self.prices.

        Returns
        -------
        pd.DataFrame
            Colonne: model, ic_mean, ic_std, ic_ir, weight_avg,
                     attributed_pnl_bps.
        """
        if forward_returns is None:
            if self.prices.empty:
                forward_returns = pd.DataFrame()
            else:
                forward_returns = self.prices.pct_change().shift(-1)

        rows = []
        for model_name, signals_df in model_signals.items():
            if signals_df.empty or forward_returns.empty:
                rows.append({
                    "model":            model_name,
                    "ic_mean":          float("nan"),
                    "ic_std":           float("nan"),
                    "ic_ir":            float("nan"),
                    "attributed_pnl_bps": float("nan"),
                })
                continue

            ic_values = []
            for d in signals_df.index:
                if d not in forward_returns.index:
                    continue
                sig = signals_df.loc[d].dropna()
                ret = forward_returns.loc[d].dropna()
                common = sig.index.intersection(ret.index)
                if len(common) < 3:
                    continue
                ic_val, _ = spearmanr(sig[common].values, ret[common].values)
                if not np.isnan(ic_val):
                    ic_values.append(ic_val)

            ic_arr = np.array(ic_values) if ic_values else np.array([float("nan")])
            ic_mean = float(np.nanmean(ic_arr))
            ic_std = float(np.nanstd(ic_arr)) if len(ic_arr) > 1 else float("nan")
            ic_ir = (
                float(ic_mean / ic_std * np.sqrt(252))
                if ic_std and ic_std > 1e-9
                else float("nan")
            )
            # P&L attribuito: IC × vol_dly × sqrt(252) × 10000 bps
            pnl_bps = (
                float(ic_mean * self.returns.std() * np.sqrt(252) * 10_000)
                if not self.returns.empty else float("nan")
            )

            rows.append({
                "model":              model_name,
                "ic_mean":            ic_mean,
                "ic_std":             ic_std,
                "ic_ir":              ic_ir,
                "attributed_pnl_bps": pnl_bps,
            })

        return pd.DataFrame(
            rows,
            columns=["model", "ic_mean", "ic_std", "ic_ir", "attributed_pnl_bps"],
        )

    # ===========================================================================
    # HTML Report
    # ===========================================================================

    def generate_html_report(self, output_path: str = "output/backtest_report.html") -> None:
        """Genera un report HTML completo con grafici Plotly interattivi.

        Grafici inclusi
        ---------------
        1. Equity curve vs SPY benchmark
        2. Drawdown chart
        3. Rolling Sharpe (252 giorni)
        4. Heatmap rendimenti mensili (calmap-style)
        5. Sommario metriche

        Parameters
        ----------
        output_path : str
            Percorso del file HTML di output.
        """
        try:
            import plotly.graph_objects as go
            import plotly.subplots as sp
            _PLOTLY = True
        except ImportError:
            _PLOTLY = False

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if not _PLOTLY:
            # Fallback: report testuale
            self._generate_text_report(out_path.with_suffix(".txt"))
            return

        # Calcola metriche per il sommario
        sharpe = self.sharpe_ratio()
        mdd = self.max_drawdown()
        calmar = self.calmar_ratio()
        cagr = self._cagr()
        n_trades = len(self.fills)

        # ── Prepara i dati ────────────────────────────────────────────────────
        equity = self.equity_curve
        returns = self.returns
        drawdown_series = self._drawdown_series()
        rolling_sharpe = self._rolling_sharpe(window=252)
        monthly_returns = self._monthly_returns_table()

        # SPY benchmark (se disponibile in prices)
        spy_equity = None
        if "SPY" in self.prices.columns and not self.prices.empty:
            spy_prices = self.prices["SPY"].dropna()
            spy_equity = (
                spy_prices / spy_prices.iloc[0] * self.initial_capital
            )

        # ── Crea i subplot ────────────────────────────────────────────────────
        fig = sp.make_subplots(
            rows=4, cols=2,
            subplot_titles=[
                "Equity Curve vs Benchmark",
                "Drawdown",
                "Rolling Sharpe (252d)",
                "Rendimenti Mensili (%)",
                "Distribuzione Rendimenti Giornalieri",
                "Turnover Giornaliero",
            ],
            specs=[
                [{"colspan": 2}, None],
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "heatmap"}],
                [{"type": "histogram"}, {"type": "bar"}],
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.06,
        )
        fig.update_layout(
            title_text=(
                f"MLCouncil Backtest Report — "
                f"Sharpe: {sharpe:.2f} | MaxDD: {mdd:.1%} | "
                f"Calmar: {calmar:.2f} | CAGR: {cagr:.1%}"
            ),
            height=1200,
            showlegend=True,
            template="plotly_dark",
        )

        # 1. Equity curve
        if not equity.empty:
            fig.add_trace(
                go.Scatter(
                    x=equity.index, y=equity.values,
                    name="MLCouncil", line=dict(color="#00CC96", width=2),
                ),
                row=1, col=1,
            )
        if spy_equity is not None:
            spy_idx = pd.to_datetime(spy_equity.index)
            fig.add_trace(
                go.Scatter(
                    x=spy_idx, y=spy_equity.values,
                    name="SPY", line=dict(color="#636EFA", width=1.5, dash="dot"),
                ),
                row=1, col=1,
            )

        # 2. Drawdown
        if not drawdown_series.empty:
            fig.add_trace(
                go.Scatter(
                    x=drawdown_series.index, y=drawdown_series.values * 100,
                    fill="tozeroy", name="Drawdown (%)",
                    line=dict(color="#EF553B"),
                ),
                row=2, col=1,
            )

        # 3. Rolling Sharpe
        if not rolling_sharpe.empty:
            fig.add_trace(
                go.Scatter(
                    x=rolling_sharpe.index, y=rolling_sharpe.values,
                    name="Rolling Sharpe (252d)",
                    line=dict(color="#FFA15A"),
                ),
                row=2, col=2,
            )
            fig.add_hline(y=0, line_dash="dash", line_color="white", row=2, col=2)

        # 4. Heatmap mensile
        if not monthly_returns.empty:
            fig.add_trace(
                go.Heatmap(
                    z=monthly_returns.values * 100,
                    x=monthly_returns.columns.astype(str).tolist(),
                    y=monthly_returns.index.astype(str).tolist(),
                    colorscale="RdYlGn",
                    zmid=0,
                    name="Rendimenti Mensili (%)",
                    text=[[f"{v:.1f}%" if not np.isnan(v) else "" for v in row]
                          for row in monthly_returns.values * 100],
                    texttemplate="%{text}",
                    showscale=True,
                ),
                row=3, col=2,
            )

        # 5. Distribuzione rendimenti
        if not returns.empty:
            fig.add_trace(
                go.Histogram(
                    x=returns.values * 100,
                    name="Rendimenti (%)",
                    nbinsx=50,
                    marker_color="#AB63FA",
                ),
                row=4, col=1,
            )

        # 6. Turnover
        to_df = self.turnover_analysis()
        if not to_df.empty and "turnover_pct" in to_df.columns:
            fig.add_trace(
                go.Bar(
                    x=to_df.index, y=to_df["turnover_pct"] * 100,
                    name="Turnover (%)", marker_color="#19D3F3",
                ),
                row=4, col=2,
            )

        # Aggiungi annotation con metriche nel titolo subplot 4
        fig.add_annotation(
            text=(
                f"Trades: {n_trades} | "
                f"CAGR: {cagr:.1%} | "
                f"Sharpe: {sharpe:.2f} | "
                f"MaxDD: {mdd:.1%}"
            ),
            xref="paper", yref="paper",
            x=0.5, y=1.02,
            showarrow=False,
            font=dict(size=11, color="white"),
        )

        # Scrivi HTML
        html_content = fig.to_html(full_html=True, include_plotlyjs="cdn")
        out_path.write_text(html_content, encoding="utf-8")
        print(f"Report salvato in: {out_path.resolve()}")

    # ===========================================================================
    # Private helpers
    # ===========================================================================

    def _compute_equity_curve(self) -> pd.Series:
        """Ricostruisce l'equity curve dai fill."""
        if self.fills.empty:
            return pd.Series([self.initial_capital], dtype=float, name="equity")

        df = self.fills.copy()
        ts_col = next(
            (c for c in ["ts_event", "timestamp", "ts_init"] if c in df.columns), None
        )
        pnl_col = next(
            (c for c in ["realized_pnl", "pnl"] if c in df.columns), None
        )

        if ts_col is None:
            return pd.Series([self.initial_capital], dtype=float, name="equity")

        df["_date"] = pd.to_datetime(df[ts_col], unit="ns", utc=True).dt.date

        if pnl_col and df[pnl_col].notna().any():
            daily_pnl = df.groupby("_date")[pnl_col].sum()
            equity = (self.initial_capital + daily_pnl.cumsum()).rename("equity")
        else:
            dates = sorted(df["_date"].unique())
            equity = pd.Series(
                self.initial_capital, index=dates, dtype=float, name="equity"
            )

        equity.index = pd.to_datetime(equity.index)
        return equity

    def _drawdown_series(self) -> pd.Series:
        if self.equity_curve.empty:
            return pd.Series(dtype=float)
        rolling_max = self.equity_curve.cummax()
        return (self.equity_curve - rolling_max) / rolling_max

    def _rolling_sharpe(self, window: int = 252) -> pd.Series:
        if len(self.returns) < window:
            return pd.Series(dtype=float)
        rfr_daily = self.risk_free_rate / 252
        excess = self.returns - rfr_daily
        rolling_mean = excess.rolling(window).mean()
        rolling_std  = self.returns.rolling(window).std()
        sr = rolling_mean / rolling_std.replace(0, float("nan")) * np.sqrt(252)
        return sr.dropna()

    def _cagr(self) -> float:
        if self.equity_curve.empty or self.initial_capital <= 0:
            return 0.0
        n_years = len(self.equity_curve) / 252
        if n_years <= 0:
            return 0.0
        final = float(self.equity_curve.iloc[-1])
        return float((final / self.initial_capital) ** (1 / n_years) - 1)

    def _monthly_returns_table(self) -> pd.DataFrame:
        """Crea una tabella Anno × Mese di rendimenti mensili."""
        if self.returns.empty:
            return pd.DataFrame()
        monthly = self.returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
        if monthly.empty:
            return pd.DataFrame()
        table = monthly.groupby(
            [monthly.index.year, monthly.index.month]
        ).sum().unstack(level=1)
        table.columns = [
            pd.Timestamp(2000, m, 1).strftime("%b") for m in table.columns
        ]
        return table

    def _generate_text_report(self, path: Path) -> None:
        """Fallback: genera un report testuale se Plotly non è disponibile."""
        sharpe = self.sharpe_ratio()
        mdd    = self.max_drawdown()
        calmar = self.calmar_ratio()
        cagr   = self._cagr()
        lines = [
            "=" * 60,
            "MLCouncil Backtest Report",
            "=" * 60,
            f"  Sharpe Ratio : {sharpe:.3f}",
            f"  Max Drawdown : {mdd:.2%}",
            f"  Calmar Ratio : {calmar:.3f}",
            f"  CAGR         : {cagr:.2%}",
            f"  N Trades     : {len(self.fills)}",
            "=" * 60,
        ]
        path.write_text("\n".join(lines))
        print(f"Report testuale salvato in: {path.resolve()}")
