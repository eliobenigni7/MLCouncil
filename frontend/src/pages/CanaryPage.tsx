import { useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "../api/client";
import { useCanaryFlags, useCanaryState, type CanaryFlag } from "../api/queries";
import { ConfirmDialog } from "../components/ConfirmDialog";
import { StatusBadge } from "../components/StatusBadge";
import { useState } from "react";

function useCanaryApply() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ name, enabled }: { name: string; enabled: boolean }) =>
      api<{ pending_changes: unknown[]; flags: unknown[] }>("/api/canary/apply", {
        method: "POST",
        body: JSON.stringify({ name, enabled }),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["canary-flags"] });
      qc.invalidateQueries({ queryKey: ["canary-state"] });
    },
  });
}

function useCanaryClear() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (name: string) =>
      api<{ pending_changes: unknown[]; flags: unknown[] }>("/api/canary/apply/clear", {
        method: "POST",
        body: JSON.stringify({ name }),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["canary-flags"] });
      qc.invalidateQueries({ queryKey: ["canary-state"] });
    },
  });
}

export function CanaryPage() {
  const flags = useCanaryFlags();
  const state = useCanaryState();
  const apply = useCanaryApply();
  const clear = useCanaryClear();
  const [pendingFlag, setPendingFlag] = useState<CanaryFlag | null>(null);

  if (flags.isLoading) return <div className="page-empty">Loading…</div>;
  if (flags.error) return <div className="page-empty">Canary configuration not available.</div>;
  const list = flags.data?.features ?? [];

  const pendingEntries = Object.entries(state.data?.pending_apply ?? {});
  const revertedEntries = Object.entries(state.data?.reverted_features ?? {});

  return (
    <div className="page">
      <h1>Canary</h1>
      <p className="caption">
        Shadow feature flags with additive pending overlay. Apply stages a change for the next run; the controller
        still auto-reverts when metrics stay below the floor.
      </p>

      <div>
        <h2>Flags</h2>
        <div className="table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>Flag</th>
                <th>Env</th>
                <th>Value</th>
                <th>Config</th>
                <th>State</th>
                <th>Effective</th>
                <th>Floor / min days</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {list.map((f) => (
                <tr key={f.name}>
                  <td className="mono">{f.name}</td>
                  <td className="mono muted">{f.env}</td>
                  <td className="mono">{f.value}</td>
                  <td>{f.config_enabled ? "enabled" : "disabled"}</td>
                  <td>
                    {f.reverted ? (
                      <StatusBadge state="failed" />
                    ) : f.pending_enabled != null ? (
                      <StatusBadge state={f.pending_enabled ? "queued" : "cancelled"} />
                    ) : (
                      <span className="muted">—</span>
                    )}
                  </td>
                  <td>
                    <StatusBadge state={f.effective_enabled ? "ok" : "warning"} />
                  </td>
                  <td className="mono">
                    {f.floor} / {f.min_days}
                  </td>
                  <td>
                    <span className="row" style={{ gap: 8 }}>
                      <button className="btn btn-sm" onClick={() => setPendingFlag(f)}>
                        Apply
                      </button>
                      {f.pending_enabled != null && (
                        <button className="btn btn-sm btn-ghost" onClick={() => clear.mutate(f.name)}>
                          Clear
                        </button>
                      )}
                    </span>
                  </td>
                </tr>
              ))}
              {list.length === 0 && (
                <tr>
                  <td colSpan={8} className="muted">No flags configured in config/canary.yaml.</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      <div className="grid-2">
        <div className="panel">
          <h2 className="panel-title">Pending apply (next run)</h2>
          {pendingEntries.length === 0 ? (
            <div className="muted">Nothing staged.</div>
          ) : (
            <div className="tag-list">
              {pendingEntries.map(([name, p]) => (
                <span key={name} className="tag" title={p.at}>
                  {name} → {p.enabled ? "on" : "off"}
                </span>
              ))}
            </div>
          )}
        </div>
        <div className="panel">
          <h2 className="panel-title">Reverted by controller</h2>
          {revertedEntries.length === 0 ? (
            <div className="muted">No reverts recorded.</div>
          ) : (
            <div className="stack">
              {revertedEntries.map(([name, r]) => (
                <div key={name} className="caption" style={{ margin: 0 }}>
                  <span className="mono">{name}</span> — {r.reason ?? "auto-reverted"}
                  {r.reverted_at ? ` (${new Date(r.reverted_at).toLocaleString()})` : ""}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      <ConfirmDialog
        open={pendingFlag !== null}
        title={`Apply pending change for ${pendingFlag?.name ?? ""}?`}
        body={
          pendingFlag
            ? `${pendingFlag.name} is currently ${pendingFlag.effective_enabled ? "on" : "off"}. Staging the opposite state for the next run (the controller can still revert it).`
            : undefined
        }
        confirmLabel="Apply"
        onCancel={() => setPendingFlag(null)}
        onConfirm={() => {
          if (pendingFlag) {
            apply.mutate({ name: pendingFlag.name, enabled: !pendingFlag.effective_enabled });
          }
          setPendingFlag(null);
        }}
      />
    </div>
  );
}
