import { useCanaryState, useSidebarMetrics } from "../api/queries";
import { KpiCard } from "../components/KpiCard";

export function OverviewPage() {
  const metrics = useSidebarMetrics();
  const canary = useCanaryState();

  if (metrics.isLoading) return <div className="page-empty">Loading…</div>;

  const reverted = Object.entries(canary.data?.reverted_features ?? {});
  const pending = Object.entries(canary.data?.pending_apply ?? {});

  return (
    <div className="page">
      <h1>Overview</h1>
      <p className="caption">
        Health of the paper-trading run: year-to-date metrics, regime and canary flag state.
      </p>

      {metrics.data ? (
        <div className="kpi-row">
          <KpiCard
            label="Sharpe (YTD)"
            value={metrics.data.sharpe_ytd.toFixed(2)}
            delta={{
              value: `${metrics.data.sharpe_delta >= 0 ? "+" : ""}${metrics.data.sharpe_delta.toFixed(2)} vs prev`,
              direction: metrics.data.sharpe_delta > 0 ? "up" : metrics.data.sharpe_delta < 0 ? "down" : "flat",
            }}
          />
          <KpiCard
            label="Max drawdown (YTD)"
            value={`${metrics.data.max_dd.toFixed(1)}%`}
            delta={{
              value: `${metrics.data.dd_delta >= 0 ? "+" : ""}${metrics.data.dd_delta.toFixed(2)}pp today`,
              direction: metrics.data.dd_delta > 0 ? "up" : metrics.data.dd_delta < 0 ? "down" : "flat",
            }}
          />
          <KpiCard
            label="IC 30d"
            value={metrics.data.ic_30d.toFixed(4)}
            delta={{
              value: `${metrics.data.ic_delta >= 0 ? "+" : ""}${metrics.data.ic_delta.toFixed(4)} vs prev`,
              direction: metrics.data.ic_delta > 0 ? "up" : metrics.data.ic_delta < 0 ? "down" : "flat",
            }}
          />
          <KpiCard
            label="Regime"
            value={metrics.data.regime}
            delta={{ value: `prob ${metrics.data.regime_prob.toFixed(1)}%`, direction: "flat" }}
          />
        </div>
      ) : (
        <div className="page-empty">No backtest results yet — run the daily pipeline first.</div>
      )}

      <div className="panel">
        <h2 className="panel-title">Canary flags</h2>
        {canary.isLoading ? (
          <div className="muted">Loading…</div>
        ) : pending.length === 0 && reverted.length === 0 ? (
          <div className="muted">No pending or reverted canary flags. Shadow features run as configured.</div>
        ) : (
          <div className="stack">
            {pending.length > 0 && (
              <div>
                <div className="caption" style={{ marginBottom: 8 }}>
                  Pending apply (next run):
                </div>
                <div className="tag-list">
                  {pending.map(([name, state]) => (
                    <span key={name} className="tag" title={state.at}>
                      {name} → {state.enabled ? "on" : "off"}
                    </span>
                  ))}
                </div>
              </div>
            )}
            {reverted.length > 0 && (
              <div>
                <div className="caption" style={{ marginBottom: 8 }}>
                  Reverted by the canary controller:
                </div>
                <div className="tag-list">
                  {reverted.map(([name, state]) => (
                    <span key={name} className="tag" title={state.reason}>
                      {name} — {state.reason ?? "auto-reverted"}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
