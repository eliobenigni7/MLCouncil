import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../api/client";
import { KpiCard } from "../components/KpiCard";
import { ConfirmDialog } from "../components/ConfirmDialog";
import { StatusBadge } from "../components/StatusBadge";

interface PipelineStatus {
  run_id?: string | null;
  status: string;
  start_time?: string | null;
  end_time?: string | null;
  partition?: string | null;
}

export function PipelinePage() {
  const qc = useQueryClient();
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [partition, setPartition] = useState("");
  const [triggeredRun, setTriggeredRun] = useState<string | null>(null);

  const status = useQuery({
    queryKey: ["pipeline-status"],
    queryFn: () => api<PipelineStatus>("/api/pipeline/status"),
    refetchInterval: 30_000,
    retry: false,
  });

  const latestPartition = useQuery({
    queryKey: ["pipeline-latest-partition"],
    queryFn: () => api<{ partition: string | null }>("/api/pipeline/latest-partition"),
    retry: false,
  });

  const automation = useQuery({
    queryKey: ["pipeline-automation", triggeredRun],
    queryFn: () =>
      api<{
        run_id: string;
        partition?: string | null;
        status: string;
        created_at?: string | null;
        updated_at?: string | null;
        error?: string | null;
      }>(`/api/pipeline/automation/${triggeredRun}`),
    enabled: triggeredRun !== null,
    refetchInterval: 5_000,
    retry: false,
  });

  const run = useMutation({
    mutationFn: (partitionValue: string | null) =>
      api<{ run_id: string }>("/api/pipeline/run", {
        method: "POST",
        body: JSON.stringify({ partition: partitionValue }),
      }),
    onSuccess: (data) => {
      setTriggeredRun(data.run_id);
      qc.invalidateQueries({ queryKey: ["pipeline-status"] });
    },
  });

  const st = status.data;
  const latest = latestPartition.data?.partition ?? "";

  return (
    <div className="page">
      <h1>Pipeline</h1>
      <p className="caption">
        Dagster orchestration status and manual run trigger. Runs the ingest → features → signals → council chain for
        a partition date.
      </p>

      {status.isLoading ? (
        <div className="page-empty">Loading…</div>
      ) : (
        <>
          <div className="kpi-row">
            <KpiCard label="Status" value={st?.status ?? "unknown"} tone={st?.status === "SUCCESS" ? "ok" : st?.status === "FAILED" ? "error" : "warning"} />
            <KpiCard label="Last run" value={st?.run_id ?? "—"} />
            <KpiCard label="Partition" value={st?.partition ?? "—"} />
          </div>
          <div className="row">
            <StatusBadge state={st?.status ?? "unknown"} />
            <span className="muted">
              {st?.start_time ? `started ${new Date(st.start_time).toLocaleString()}` : ""}
              {st?.end_time ? ` · finished ${new Date(st.end_time).toLocaleString()}` : ""}
            </span>
          </div>
        </>
      )}

      <div className="panel">
        <h2 className="panel-title">Run pipeline</h2>
        <div className="form-grid">
          <label>
            Partition (defaults to latest)
            <input value={partition || latest} onChange={(e) => setPartition(e.target.value)} placeholder={latest || "YYYY-MM-DD"} />
          </label>
          <button className="btn btn-primary" onClick={() => setConfirmOpen(true)} disabled={run.isPending}>
            {run.isPending ? "Starting…" : "Run Pipeline"}
          </button>
        </div>
        {run.data && (
          <p className="form-status" style={{ marginTop: 12 }}>
            Started run {run.data.run_id}.
          </p>
        )}
        {run.error && <p className="form-error">{String(run.error)}</p>}

        {automation.data && (
          <div className="panel" style={{ marginTop: 14, background: "var(--bg-inset)" }}>
            <div className="panel-title">Automation task</div>
            <div className="row">
              <span className="mono">{automation.data.run_id}</span>
              <StatusBadge state={automation.data.status} />
              <span className="muted">{automation.data.partition ?? "—"}</span>
            </div>
            {automation.data.error && <p className="form-error">{automation.data.error}</p>}
          </div>
        )}
      </div>

      <ConfirmDialog
        open={confirmOpen}
        title="Trigger a pipeline run?"
        body={`Run the Dagster pipeline for ${(partition || latest || "the latest available partition")}. This refreshes daily artifacts used by every page here.`}
        confirmLabel="Confirm"
        onCancel={() => setConfirmOpen(false)}
        onConfirm={() => {
          run.mutate(partition || latest || null);
          setConfirmOpen(false);
        }}
      />
    </div>
  );
}
