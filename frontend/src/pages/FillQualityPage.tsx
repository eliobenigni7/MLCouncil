import { useFillQuality, useCostCalibration, type FillQualityRow } from "../api/queries";

function fmt(v: number | null | undefined, digits = 2): string {
  return v == null || Number.isNaN(v) ? "—" : v.toFixed(digits);
}

export function FillQualityPage() {
  const fillQuality = useFillQuality();
  const calibration = useCostCalibration();

  const rows = fillQuality.data?.records ?? [];
  const calibration404 = (calibration.error as { status?: number } | null)?.status === 404;

  return (
    <div className="page">
      <h1>Fill Quality</h1>
      <p className="caption">
        Per-ticker implementation shortfall, lookup slippage and calibrated kappa from the cost calibration track.
      </p>

      <div>
        <h2>Per-ticker summary</h2>
        {fillQuality.isLoading ? (
          <div className="page-empty">Loading…</div>
        ) : rows.length === 0 ? (
          <div className="page-empty">No fill data yet — fills are recorded under data/operations/fills/.</div>
        ) : (
          <div className="table-wrap">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Ticker</th>
                  <th className="num">Fills</th>
                  <th className="num">Median IS (bps)</th>
                  <th className="num">Lookup slippage (bps)</th>
                  <th className="num">Calibrated kappa (bps)</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r: FillQualityRow, i) => (
                  <tr key={i}>
                    <td className="mono">{r.ticker}</td>
                    <td className="num">{r.fill_count ?? "—"}</td>
                    <td className="num">{fmt(r.median_is_bps)}</td>
                    <td className="num">{fmt(r.lookup_slippage_bps)}</td>
                    <td className="num">{fmt(r.kappa_calibrated_bps)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <div>
        <h2>Calibration artifact</h2>
        {calibration.isLoading ? (
          <div className="page-empty">Loading…</div>
        ) : calibration404 ? (
          <div className="page-empty">
            No calibration artifact yet — cost calibration produces data/operations/cost_calibration.json after enough
            fills accumulate.
          </div>
        ) : calibration.data ? (
          <pre className="code">{JSON.stringify(calibration.data, null, 2)}</pre>
        ) : null}
      </div>
    </div>
  );
}
