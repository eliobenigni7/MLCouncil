import Plot from "react-plotly.js";
import { useCurrentRegime, useEquity, useRegimeHistory, useWeightsHistory } from "../api/queries";
import { currentWeightsRadar, regimeProbabilityGauge, regimeTimeline } from "../features/analytics/charts";

export function RegimePage() {
  const regime = useCurrentRegime();
  const history = useRegimeHistory();
  const equity = useEquity();
  const weights = useWeightsHistory();
  const loading = regime.isLoading || history.isLoading || equity.isLoading || weights.isLoading;
  const missing = [regime.error, history.error, equity.error, weights.error].some(
    (e) => e && (e as { status?: number }).status === 404,
  );

  if (loading) return <div className="page-empty">Loading…</div>;
  if (missing) return <div className="page-empty">No regime artifacts yet — run the daily pipeline first.</div>;
  if (!regime.data || !history.data || !equity.data || !weights.data) return null;

  const latestWeights = weights.data.records[weights.data.records.length - 1] ?? {};
  const radarWeights: Record<string, number> = {};
  for (const model of ["lgbm", "sentiment", "hmm"] as const) {
    const w = latestWeights[model];
    if (typeof w === "number") radarWeights[model] = w;
  }

  return (
    <div className="page">
      <h1>Regime</h1>
      <p className="caption">HMM regime classification with probabilities, timeline and current council weights.</p>
      <div className="plot-card">
        <Plot
          {...regimeProbabilityGauge({
            bull: regime.data.bull,
            bear: regime.data.bear,
            transition: regime.data.transition,
          })}
          style={{ width: "100%", height: 300 }}
          useResizeHandler
        />
      </div>
      <div className="plot-card">
        <Plot {...regimeTimeline(history.data.records, equity.data)} style={{ width: "100%", height: 420 }} useResizeHandler />
      </div>
      <div className="plot-card">
        <Plot {...currentWeightsRadar(radarWeights)} style={{ width: "100%", height: 400 }} useResizeHandler />
      </div>
    </div>
  );
}
