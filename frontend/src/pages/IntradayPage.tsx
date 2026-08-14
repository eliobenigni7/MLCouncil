import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../api/client";
import { KpiCard } from "../components/KpiCard";
import { ConfirmDialog } from "../components/ConfirmDialog";
import { StatusBadge } from "../components/StatusBadge";

interface IntradayStatus {
  running: boolean;
  paused: boolean;
  schedule_minutes: number;
  market_session: string;
  last_completed_slot?: string | null;
  latest_decision_id?: string | null;
}

interface Decision {
  decision_id: string;
  as_of: string;
  market_session: string;
  agent_trace: Record<string, unknown>;
  execution_intents: Record<string, unknown>[];
}

export function IntradayPage() {
  const qc = useQueryClient();
  const [confirmExecute, setConfirmExecute] = useState(false);

  const status = useQuery({
    queryKey: ["intraday-status"],
    queryFn: () => api<IntradayStatus>("/api/intraday/status"),
    refetchInterval: 15_000,
    retry: false,
  });

  const decision = useQuery({
    queryKey: ["intraday-latest-decision"],
    queryFn: () => api<Decision>("/api/intraday/decisions/latest"),
    retry: false,
  });

  const explain = useQuery({
    queryKey: ["intraday-explain", decision.data?.decision_id ?? null],
    queryFn: () =>
      api<Record<string, unknown>>(`/api/intraday/decisions/${decision.data?.decision_id}/explain`),
    enabled: decision.data?.decision_id != null,
    retry: false,
  });

  const control = (action: "start" | "pause" | "resume" | "stop") =>
    api<IntradayStatus>(`/api/intraday/control/${action}`, { method: "POST" });

  const controlMutation = useMutation({
    mutationFn: control,
    onSuccess: () => qc.invalidateQueries({ queryKey: ["intraday-status"] }),
  });

  const cycle = useMutation({
    mutationFn: () => api<unknown>("/api/intraday/cycle", { method: "POST" }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["intraday-status"] });
      qc.invalidateQueries({ queryKey: ["intraday-latest-decision"] });
    },
  });

  const executeDecision = useMutation({
    mutationFn: (decisionId: string) =>
      api<unknown>(`/api/intraday/decisions/${decisionId}/execute`, { method: "POST" }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["intraday-latest-decision"] }),
  });

  const st = status.data;
  const d = decision.data;

  return (
    <div className="page">
      <h1>Intraday</h1>
      <p className="caption">
        Intraday supervisor runtime: periodic decision cycles on the 15-minute schedule, with explainable agent traces.
      </p>

      {status.isLoading ? (
        <div className="page-empty">Loading…</div>
      ) : (
        <>
          <div className="kpi-row">
            <KpiCard
              label="Supervisor"
              value={st?.running ? "Running" : "Stopped"}
              tone={st?.running ? (st?.paused ? "warning" : "ok") : "error"}
              delta={{ value: st?.paused ? "paused" : st?.running ? "active" : "idle", direction: "flat" }}
            />
            <KpiCard label="Market session" value={st?.market_session ?? "—"} />
            <KpiCard label="Schedule" value={`${st?.schedule_minutes ?? "—"} min`} />
            <KpiCard label="Last slot" value={st?.last_completed_slot ?? "—"} />
          </div>
          <div className="row">
            <button className="btn btn-primary" onClick={() => controlMutation.mutate("start")}
              disabled={Boolean(st?.running) || controlMutation.isPending}>
              Start
            </button>
            <button className="btn" onClick={() => controlMutation.mutate("pause")}
              disabled={!st?.running || Boolean(st?.paused) || controlMutation.isPending}>
              Pause
            </button>
            <button className="btn" onClick={() => controlMutation.mutate("resume")}
              disabled={!st?.running || !st?.paused || controlMutation.isPending}>
              Resume
            </button>
            <button className="btn btn-danger" onClick={() => controlMutation.mutate("stop")}
              disabled={!st?.running || controlMutation.isPending}>
              Stop
            </button>
            <button className="btn" onClick={() => cycle.mutate()} disabled={cycle.isPending}>
              {cycle.isPending ? "Cycling…" : "Run cycle now"}
            </button>
          </div>
        </>
      )}

      <div>
        <h2>Latest decision</h2>
        {decision.isLoading ? (
          <div className="page-empty">Loading…</div>
        ) : d ? (
          <div className="stack">
            <div className="row">
              <StatusBadge state={d.market_session === "open" ? "running" : "cancelled"} />
              <span className="mono">{d.decision_id}</span>
              <span className="muted">{d.as_of}</span>
              <button className="btn btn-sm btn-primary" onClick={() => setConfirmExecute(true)}>
                Execute decision
              </button>
            </div>
            <div className="grid-2">
              {d.execution_intents.length > 0 && (
                <div className="table-wrap">
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Ticker</th>
                        <th>Side</th>
                        <th className="num">Notional</th>
                      </tr>
                    </thead>
                    <tbody>
                      {d.execution_intents.map((intent, i) => (
                        <tr key={i}>
                          <td className="mono">{String(intent.symbol ?? intent.ticker ?? "—")}</td>
                          <td>{String(intent.side ?? "—").toUpperCase()}</td>
                          <td className="num">{String(intent.notional ?? intent.qty ?? "—")}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
              {explain.data && (
                <div>
                  <div className="caption" style={{ marginBottom: 8 }}>Agent trace</div>
                  <pre className="code">{JSON.stringify(explain.data, null, 2)}</pre>
                </div>
              )}
            </div>
          </div>
        ) : (
          <div className="page-empty">No intraday decisions yet — start the supervisor or run a cycle.</div>
        )}
      </div>

      <ConfirmDialog
        open={confirmExecute}
        title="Execute latest decision?"
        body="Sends the execution intents of the latest decision to the paper broker."
        confirmLabel="Confirm"
        onCancel={() => setConfirmExecute(false)}
        onConfirm={() => {
          if (d) executeDecision.mutate(d.decision_id);
          setConfirmExecute(false);
        }}
      />
    </div>
  );
}
