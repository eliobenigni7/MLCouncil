import { useQuery } from "@tanstack/react-query";
import { api } from "../api/client";
import { StatusBadge } from "../components/StatusBadge";

interface Manifest {
  schema_version?: number;
  updated_at?: string;
  updated_by?: string;
  models?: Record<string, { family?: string; checkpoint?: string }>;
  council?: Record<string, unknown>;
  experts?: Record<string, { enabled?: boolean; checkpoint?: string }>;
  features?: Record<string, unknown>;
  promotion_history?: { model?: string; at?: string; gate_report?: string; by?: string }[];
}

interface PromotionReport {
  model?: string;
  promotion_passed?: boolean;
  champion_metrics?: { oos_sharpe?: number; pbo?: number; walk_forward_window_count?: number };
  challenger_metrics?: { oos_sharpe?: number; pbo?: number; walk_forward_window_count?: number };
}

interface Streak {
  consecutive_passes?: number;
  auto_promote_eligible?: boolean;
}

export function PromotionPage() {
  const manifest = useQuery({
    queryKey: ["promotion-manifest"],
    queryFn: () => api<Manifest>("/api/promotion/manifest"),
  });
  const reports = useQuery({
    queryKey: ["promotion-reports"],
    queryFn: () => api<{ reports: Record<string, PromotionReport>; streaks: Record<string, Streak> }>("/api/promotion/reports"),
  });
  const artifacts = useQuery({
    queryKey: ["promotion-artifacts"],
    queryFn: () =>
      api<{ artifacts: { path: string; exists: boolean; mtime: number | null }[] }>("/api/promotion/shadow-artifacts"),
  });

  const manifest404 = (manifest.error as { status?: number } | null)?.status === 404;
  const loading = manifest.isLoading || reports.isLoading || artifacts.isLoading;

  if (loading) return <div className="page-empty">Loading…</div>;

  const reportRows = Object.entries(reports.data?.reports ?? {});
  const artifactRows = artifacts.data?.artifacts ?? [];

  return (
    <div className="page">
      <h1>Promotion</h1>
      <p className="caption">
        Production manifest, walk-forward promotion gate reports and shadow artifact status.
      </p>

      <div className="panel">
        <h2 className="panel-title">Production manifest</h2>
        {manifest404 || !manifest.data ? (
          <div className="muted">
            No production manifest yet — promote a model first (scripts/promote_model.py).
          </div>
        ) : (
          <div className="stack">
            <div className="row">
              <span className="tag">schema {manifest.data.schema_version ?? "?"}</span>
              <span className="tag">{manifest.data.updated_at ?? "—"}</span>
              <span className="tag">by {manifest.data.updated_by ?? "—"}</span>
            </div>
            <div className="grid-2">
              <div>
                <div className="caption" style={{ marginBottom: 8 }}>Models</div>
                <div className="table-wrap">
                  <table className="data-table">
                    <thead>
                      <tr><th>Role</th><th>Family</th><th>Checkpoint</th></tr>
                    </thead>
                    <tbody>
                      {Object.entries(manifest.data.models ?? {}).map(([role, m]) => (
                        <tr key={role}>
                          <td>{role}</td>
                          <td className="mono">{m?.family ?? "—"}</td>
                          <td className="mono muted">{m?.checkpoint ?? "—"}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
              <div>
                <div className="caption" style={{ marginBottom: 8 }}>Council</div>
                <div className="tag-list">
                  {Object.entries(manifest.data.council ?? {}).map(([k, v]) => (
                    <span key={k} className="tag">
                      {k}: {String(v)}
                    </span>
                  ))}
                </div>
                <div className="caption" style={{ margin: "12px 0 8px" }}>Experts</div>
                <div className="tag-list">
                  {Object.entries(manifest.data.experts ?? {}).map(([k, e]) => (
                    <span key={k} className="tag">
                      {k}: {e?.enabled ? "on" : "off"}
                    </span>
                  ))}
                </div>
              </div>
            </div>
            {(manifest.data.promotion_history?.length ?? 0) > 0 && (
              <div>
                <div className="caption" style={{ marginBottom: 8 }}>Promotion history</div>
                <div className="table-wrap">
                  <table className="data-table">
                    <thead>
                      <tr><th>Model</th><th>At</th><th>Gate report</th><th>By</th></tr>
                    </thead>
                    <tbody>
                      {manifest.data.promotion_history!.map((p, i) => (
                        <tr key={i}>
                          <td className="mono">{p.model ?? "—"}</td>
                          <td className="mono muted">{p.at ?? "—"}</td>
                          <td className="mono muted">{p.gate_report ?? "—"}</td>
                          <td className="mono muted">{p.by ?? "—"}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      <div>
        <h2>Walk-forward gate reports</h2>
        {reportRows.length === 0 ? (
          <div className="page-empty">No gate reports yet — run the walk-forward CI first.</div>
        ) : (
          <div className="table-wrap">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Promotion passed</th>
                  <th>Consecutive passes</th>
                  <th>Auto-promote eligible</th>
                  <th className="num">Champion OOS Sharpe</th>
                  <th className="num">Challenger OOS Sharpe</th>
                </tr>
              </thead>
              <tbody>
                {reportRows.map(([model, r]) => {
                  const streak = reports.data?.streaks?.[model];
                  return (
                    <tr key={model}>
                      <td className="mono">{r.model ?? model}</td>
                      <td>
                        <StatusBadge state={r.promotion_passed ? "ok" : "failed"} />
                      </td>
                      <td className="num">{streak?.consecutive_passes ?? 0}</td>
                      <td>
                        {streak?.auto_promote_eligible ? (
                          <StatusBadge state="ok" />
                        ) : (
                          <span className="muted">—</span>
                        )}
                      </td>
                      <td className="num">{r.champion_metrics?.oos_sharpe?.toFixed(3) ?? "—"}</td>
                      <td className="num">{r.challenger_metrics?.oos_sharpe?.toFixed(3) ?? "—"}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <div>
        <h2>Shadow artifacts</h2>
        <div className="table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>Artifact</th>
                <th>Present</th>
                <th>Last modified</th>
              </tr>
            </thead>
            <tbody>
              {artifactRows.map((a) => (
                <tr key={a.path}>
                  <td className="mono">{a.path}</td>
                  <td>
                    <StatusBadge state={a.exists ? "ok" : "failed"} />
                  </td>
                  <td className="mono muted">
                    {a.mtime ? new Date(a.mtime * 1000).toLocaleString() : "—"}
                  </td>
                </tr>
              ))}
              {artifactRows.length === 0 && (
                <tr>
                  <td colSpan={3} className="muted">No shadow artifacts tracked.</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
