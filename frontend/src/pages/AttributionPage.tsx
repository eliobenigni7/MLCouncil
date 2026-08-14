import Plot from "react-plotly.js";
import { useAttribution, useIcHistory, useWeightsHistory } from "../api/queries";
import { icRollingChart, modelContributionBar, weightEvolutionChart } from "../features/analytics/charts";

export function AttributionPage() {
  const attribution = useAttribution();
  const icHistory = useIcHistory();
  const weights = useWeightsHistory();
  const loading = attribution.isLoading || icHistory.isLoading || weights.isLoading;
  const missing = [attribution.error, icHistory.error, weights.error].some(
    (e) => e && (e as { status?: number }).status === 404,
  );

  if (loading) return <div className="page-empty">Loading…</div>;
  if (missing) return <div className="page-empty">No attribution artifacts yet — run the daily pipeline first.</div>;
  if (!attribution.data || !icHistory.data || !weights.data) return null;

  const records = attribution.data.records;

  return (
    <div className="page">
      <h1>Attribution</h1>
      <p className="caption">
        Per-model P&L contribution, rolling IC and weight evolution of the council aggregator.
      </p>
      <div className="plot-card">
        <Plot {...modelContributionBar(records)} style={{ width: "100%", height: 320 }} useResizeHandler />
      </div>
      <div className="plot-card">
        <Plot {...icRollingChart(icHistory.data.records)} style={{ width: "100%", height: 340 }} useResizeHandler />
      </div>
      <div className="plot-card">
        <Plot {...weightEvolutionChart(weights.data.records)} style={{ width: "100%", height: 340 }} useResizeHandler />
      </div>
    </div>
  );
}
