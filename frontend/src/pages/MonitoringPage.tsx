import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../api/client";
import { ConfirmDialog } from "../components/ConfirmDialog";
import { StatusBadge } from "../components/StatusBadge";

interface AlertRow {
  severity?: string;
  level?: string;
  model_name?: string;
  check_type?: string;
  message?: string;
  timestamp?: string;
}

interface SettingField {
  key: string;
  label: string;
  description?: string;
  secret?: boolean;
  placeholder?: string;
  value?: string | null;
  configured?: boolean;
}

interface HealthSignal {
  level?: string;
  value?: unknown;
  threshold?: unknown;
  note?: string;
}

export function MonitoringPage() {
  const qc = useQueryClient();
  const [values, setValues] = useState<Record<string, string>>({});
  const [confirmOpen, setConfirmOpen] = useState(false);

  const alerts = useQuery({
    queryKey: ["monitoring-alerts"],
    queryFn: () => api<AlertRow[]>("/api/monitoring/alerts"),
    refetchInterval: 60_000,
    retry: false,
  });

  const history = useQuery({
    queryKey: ["monitoring-alert-history"],
    queryFn: () => api<AlertRow[]>("/api/monitoring/alerts/history?limit=30"),
    retry: false,
  });

  const health = useQuery({
    queryKey: ["monitoring-health"],
    queryFn: () => api<Record<string, HealthSignal>>("/api/monitoring/health"),
    retry: false,
  });

  const settings = useQuery({
    queryKey: ["monitoring-settings"],
    queryFn: () => api<{ path: string; settings: SettingField[] }>("/api/monitoring/settings"),
    retry: false,
  });

  const save = useMutation({
    mutationFn: (payload: Record<string, string>) =>
      api<unknown>("/api/monitoring/settings", { method: "PUT", body: JSON.stringify({ values: payload }) }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["monitoring-settings"] });
      qc.invalidateQueries({ queryKey: ["monitoring-alerts"] });
    },
  });

  const alertList = alerts.data ?? [];
  const historyList = history.data ?? [];
  const healthEntries = Object.entries(health.data ?? {});
  const settingFields = settings.data?.settings ?? [];

  return (
    <div className="page">
      <h1>Monitoring</h1>
      <p className="caption">
        Active and historical alerts, immune-system health signals and runtime environment settings.
      </p>

      <div>
        <h2>Active alerts</h2>
        {alerts.isLoading ? (
          <div className="page-empty">Loading…</div>
        ) : alertList.length === 0 ? (
          <div className="page-empty" style={{ padding: "24px 16px" }}>No active alerts.</div>
        ) : (
          <div className="table-wrap">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Severity</th>
                  <th>Model</th>
                  <th>Check</th>
                  <th>Message</th>
                  <th>Timestamp</th>
                </tr>
              </thead>
              <tbody>
                {alertList.map((a, i) => (
                  <tr key={i}>
                    <td><StatusBadge state={a.severity ?? a.level ?? "unknown"} /></td>
                    <td className="mono">{a.model_name ?? "—"}</td>
                    <td className="mono">{a.check_type ?? "—"}</td>
                    <td>{a.message ?? "—"}</td>
                    <td className="muted">{a.timestamp ?? "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <div>
        <h2>Health signals</h2>
        {health.isLoading ? (
          <div className="page-empty">Loading…</div>
        ) : healthEntries.length === 0 ? (
          <div className="page-empty" style={{ padding: "24px 16px" }}>No health signals collected yet.</div>
        ) : (
          <div className="table-wrap">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Signal</th>
                  <th>Level</th>
                  <th className="num">Value</th>
                  <th className="num">Threshold</th>
                  <th>Note</th>
                </tr>
              </thead>
              <tbody>
                {healthEntries.map(([name, s]) => (
                  <tr key={name}>
                    <td className="mono">{name}</td>
                    <td><StatusBadge state={s.level ?? "unknown"} /></td>
                    <td className="num">{String(s.value ?? "—")}</td>
                    <td className="num">{String(s.threshold ?? "—")}</td>
                    <td className="muted">{s.note ?? "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <div>
        <h2>Runtime settings</h2>
        {settings.isLoading ? (
          <div className="page-empty">Loading…</div>
        ) : settingFields.length === 0 ? (
          <div className="page-empty" style={{ padding: "24px 16px" }}>No settings defined.</div>
        ) : (
          <div className="panel">
            <p className="form-status" style={{ marginBottom: 14 }}>
              Shared file: <span className="mono">{settings.data?.path ?? "—"}</span>
            </p>
            <div className="form-grid">
              {settingFields.map((f) => {
                const immutable = f.key === "MLCOUNCIL_API_KEY";
                return (
                  <label key={f.key}>
                    {f.label} {immutable && <span>(immutable)</span>}
                    <input
                      type={f.secret ? "password" : "text"}
                      autoComplete="off"
                      placeholder={f.placeholder ?? ""}
                      disabled={immutable}
                      defaultValue={f.value ?? ""}
                      onChange={(e) => setValues((v) => ({ ...v, [f.key]: e.target.value }))}
                    />
                    {f.description && <span>{f.description}</span>}
                  </label>
                );
              })}
            </div>
            <div className="form-actions">
              <button className="btn btn-primary" onClick={() => setConfirmOpen(true)} disabled={save.isPending}>
                {save.isPending ? "Saving…" : "Save settings"}
              </button>
              {save.error && <p className="form-error">{String(save.error)}</p>}
            </div>
          </div>
        )}
      </div>

      <div>
        <h2>Alert history (30)</h2>
        {history.isLoading ? (
          <div className="page-empty">Loading…</div>
        ) : historyList.length === 0 ? (
          <div className="page-empty" style={{ padding: "24px 16px" }}>No alert history.</div>
        ) : (
          <div className="table-wrap">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Severity</th>
                  <th>Model</th>
                  <th>Check</th>
                  <th>Message</th>
                  <th>Timestamp</th>
                </tr>
              </thead>
              <tbody>
                {historyList.map((a, i) => (
                  <tr key={i}>
                    <td><StatusBadge state={a.severity ?? a.level ?? "unknown"} /></td>
                    <td className="mono">{a.model_name ?? "—"}</td>
                    <td className="mono">{a.check_type ?? "—"}</td>
                    <td>{a.message ?? "—"}</td>
                    <td className="muted">{a.timestamp ?? "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <ConfirmDialog
        open={confirmOpen}
        title="Save runtime settings?"
        body="Writes the values to the shared runtime env file. Secret fields left at their masked value are kept untouched by the backend."
        confirmLabel="Confirm"
        onCancel={() => setConfirmOpen(false)}
        onConfirm={() => {
          save.mutate(values);
          setConfirmOpen(false);
        }}
      />
    </div>
  );
}
