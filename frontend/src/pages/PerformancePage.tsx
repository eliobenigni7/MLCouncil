import Plot from "react-plotly.js";
import { useBenchmark, useDailyReturns, useEquity } from "../api/queries";
import { drawdownChart, equityCurveChart, monthlyReturnsHeatmap, rollingSharpeChart } from "../features/analytics/charts";
import { KpiCard } from "../components/KpiCard";

export function PerformancePage() {
  const equity = useEquity();
  const benchmark = useBenchmark();
  const returns = useDailyReturns();
  const loading = equity.isLoading || benchmark.isLoading || returns.isLoading;
  const missing = [equity.error, benchmark.error, returns.error].some(
    (e) => e && (e as { status?: number }).status === 404,
  );

  if (loading) return <div className="page-empty">Loading…</div>;
  if (missing) return <div className="page-empty">No backtest results yet — run the daily pipeline first.</div>;
  if (!equity.data || !benchmark.data || !returns.data) return null;

  const eq = equity.data;
  const finalEquity = eq.values[eq.values.length - 1];
  const cagr = Math.pow(finalEquity / 100, 365 / Math.max(eq.dates.length, 1)) - 1;
  const dd = drawdownChart(eq).data[0].y as number[];
  const maxDd = Math.min(...dd);

  return (
    <div className="page">
      <h1>Performance</h1>
      <p className="caption">
        Equity curve, drawdown, rolling Sharpe and monthly returns for the paper-trading run, normalized to base 100.
      </p>
      <div className="kpi-row">
        <KpiCard label="Final equity (base 100)" value={finalEquity.toFixed(1)} />
        <KpiCard label="Max drawdown" value={`${maxDd.toFixed(1)}%`} />
        <KpiCard label="CAGR" value={`${(cagr * 100).toFixed(1)}%`} />
      </div>
      <div className="plot-card">
        <Plot {...equityCurveChart(eq, benchmark.data)} style={{ width: "100%", height: 420 }} useResizeHandler />
      </div>
      <div className="plot-card">
        <Plot {...drawdownChart(eq)} style={{ width: "100%", height: 300 }} useResizeHandler />
      </div>
      <div className="plot-card">
        <Plot {...rollingSharpeChart(returns.data)} style={{ width: "100%", height: 300 }} useResizeHandler />
      </div>
      <div className="plot-card">
        <Plot {...monthlyReturnsHeatmap(returns.data)} style={{ width: "100%", height: 360 }} useResizeHandler />
      </div>
    </div>
  );
}
