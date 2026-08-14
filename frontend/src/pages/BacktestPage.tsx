import { useMemo, useState } from "react";
import { useQuery, useQueries } from "@tanstack/react-query";
import { api } from "../api/client";
import { useCancelJob, useJobs, useSubmitBacktest, type BacktestForm, type JobEntry } from "../features/experiments/jobs";
import { StatusBadge } from "../components/StatusBadge";
import { DataTable } from "../components/DataTable";
import Plot from "react-plotly.js";
import { playgroundOverlayChart, type Series } from "../features/analytics/charts";

const DEFAULTS: BacktestForm = {
  start_date: "2024-01-01",
  end_date: "2025-01-01",
  universe: ["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
  initial_capital: 100000,
  slippage_bps: 3,
  commission_bps: 0.5,
  regime_weights: {
    bull: { lgbm: 0.55, sentiment: 0.25, hmm: 0.2 },
    bear: { lgbm: 0.35, sentiment: 0.15, hmm: 0.5 },
    transition: { lgbm: 0.45, sentiment: 0.2, hmm: 0.35 },
  },
  weight_clip_min: 0.05,
  weight_clip_max: 0.6,
  ic_rolling_window: 60,
  sharpe_rolling_window: 120,
  use_orthogonality: true,
  max_correlation: 0.65,
  max_position: 0.08,
  max_turnover: 0.2,
  max_vol_ann: 0.3,
  sector_cap: 0.45,
  min_signal_strength: 0.2,
  note: "",
};

const REGIMES = ["bull", "bear", "transition"] as const;
const MODELS = ["lgbm", "sentiment", "hmm"] as const;

function normalizeWeights(weights: Record<string, number>): Record<string, number> {
  const total = Object.values(weights).reduce((a, b) => a + Math.max(0, b), 0);
  if (total <= 1e-9) return Object.fromEntries(Object.keys(weights).map((k) => [k, 1 / Math.max(Object.keys(weights).length, 1)]));
  return Object.fromEntries(Object.entries(weights).map(([k, v]) => [k, Math.max(0, v) / total]));
}

interface SnapshotRecord {
  timestamp?: string;
  start_date?: string;
  end_date?: string;
  n_tickers?: number | null;
  sharpe?: number | null;
  max_drawdown?: number | null;
  cagr?: number | null;
  final_equity?: number | null;
  note?: string;
  path: string;
}

interface SnapshotDetail {
  snapshot_dir: string;
  equity: Series;
  params: Record<string, unknown>;
}

function SnapshotOverlay() {
  const snapshots = useQuery({
    queryKey: ["snapshots"],
    queryFn: () => api<{ snapshots: SnapshotRecord[] }>("/api/experiments/snapshots"),
  });
  const [selected, setSelected] = useState<string[]>([]);

  const detailQueries = useQueries({
    queries: selected.map((dir) => ({
      queryKey: ["snapshot", dir],
      queryFn: () => api<SnapshotDetail>(`/api/experiments/snapshots/${encodeURIComponent(dir)}`),
    })),
  });

  const curves = useMemo(() => {
    const out: Record<string, Series> = {};
    selected.forEach((dir, i) => {
      const equity = detailQueries[i]?.data?.equity;
      if (equity && equity.dates.length > 0) out[dir] = equity;
    });
    return out;
  }, [selected, detailQueries]);

  if (snapshots.isLoading) return <div className="page-empty">Loading…</div>;
  const list = snapshots.data?.snapshots ?? [];

  if (list.length === 0) {
    return <div className="page-empty">No snapshots yet — run a backtest to create one under data/results_playground/.</div>;
  }

  return (
    <div className="stack">
      <div className="table-wrap">
        <table className="data-table">
          <thead>
            <tr>
              <th>Compare</th>
              <th>Run</th>
              <th>Window</th>
              <th>Tickers</th>
              <th className="num">Sharpe</th>
              <th className="num">Max DD</th>
              <th className="num">Final equity</th>
              <th>Note</th>
            </tr>
          </thead>
          <tbody>
            {list.map((s) => {
              const checked = selected.includes(s.path);
              return (
                <tr key={s.path}>
                  <td>
                    <input
                      type="checkbox"
                      checked={checked}
                      onChange={(e) =>
                        setSelected((prev) =>
                          e.target.checked ? [...prev, s.path] : prev.filter((p) => p !== s.path),
                        )
                      }
                      aria-label={`compare ${s.timestamp ?? s.path}`}
                    />
                  </td>
                  <td className="mono">{s.timestamp}</td>
                  <td className="mono">{s.start_date} → {s.end_date}</td>
                  <td>{s.n_tickers ?? "—"}</td>
                  <td className="num">{s.sharpe != null ? s.sharpe.toFixed(2) : "—"}</td>
                  <td className="num">{s.max_drawdown != null ? `${(s.max_drawdown * 100).toFixed(1)}%` : "—"}</td>
                  <td className="num">{s.final_equity != null ? s.final_equity.toFixed(0) : "—"}</td>
                  <td className="muted">{s.note || "—"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {Object.keys(curves).length > 0 && (
        <div className="plot-card">
          <Plot {...playgroundOverlayChart(curves)} style={{ width: "100%", height: 420 }} useResizeHandler />
        </div>
      )}
    </div>
  );
}

function RangeField({
  label,
  value,
  min,
  max,
  step,
  disabled,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  disabled?: boolean;
  onChange: (v: number) => void;
}) {
  return (
    <label className="range-field">
      <span>
        {label} <span className="range-value mono">{value.toFixed(2)}</span>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </label>
  );
}

export function BacktestPage() {
  const jobs = useJobs();
  const submit = useSubmitBacktest();
  const cancel = useCancelJob();
  const [form, setForm] = useState<BacktestForm>(DEFAULTS);
  const [universeText, setUniverseText] = useState(DEFAULTS.universe.join(", "));

  const set = <K extends keyof BacktestForm>(key: K, value: BacktestForm[K]) =>
    setForm((f) => ({ ...f, [key]: value }));

  const setRegimeWeight = (regime: string, model: string, value: number) =>
    setForm((f) => ({
      ...f,
      regime_weights: {
        ...f.regime_weights,
        [regime]: { ...f.regime_weights[regime], [model]: value },
      },
    }));

  async function onRun() {
    const universe = universeText
      .split(",")
      .map((t) => t.trim().toUpperCase())
      .filter(Boolean);
    const regimeWeights = Object.fromEntries(
      REGIMES.map((r) => [r, normalizeWeights(form.regime_weights[r])]),
    ) as BacktestForm["regime_weights"];
    await submit.mutateAsync({ ...form, universe, regime_weights: regimeWeights });
  }

  const columns = ["id", "state", "created_at", "snapshot_path", "error"];
  return (
    <div className="page">
      <h1>Backtest Playground</h1>
      <p className="caption">
        Ad-hoc backtest without Alpaca: deterministic proxy signals, 5-day rebalance, snapshots persisted under
        data/results_playground/.
      </p>

      <div className="panel form-section">
        <h2 className="panel-title">Run parameters</h2>
        <div className="form-grid">
          <label>
            Start
            <input type="date" value={form.start_date} onChange={(e) => set("start_date", e.target.value)} />
          </label>
          <label>
            End
            <input type="date" value={form.end_date} onChange={(e) => set("end_date", e.target.value)} />
          </label>
          <label>
            Initial capital (USD)
            <input
              type="number"
              min={1000}
              max={10_000_000}
              step={10_000}
              value={form.initial_capital}
              onChange={(e) => set("initial_capital", Number(e.target.value))}
            />
          </label>
          <label>
            Slippage (bps)
            <input
              type="number"
              min={0}
              max={200}
              step={0.5}
              value={form.slippage_bps}
              onChange={(e) => set("slippage_bps", Number(e.target.value))}
            />
          </label>
          <label>
            Commission (bps)
            <input
              type="number"
              min={0}
              max={50}
              step={0.1}
              value={form.commission_bps}
              onChange={(e) => set("commission_bps", Number(e.target.value))}
            />
          </label>
        </div>

        <label>
          Universe (comma-separated tickers)
          <input value={universeText} onChange={(e) => setUniverseText(e.target.value)} />
        </label>

        <div>
          <div className="caption" style={{ marginBottom: 10 }}>
            Council — regime weights (normalized to sum 1 on submit)
          </div>
          <div className="grid-2">
            {REGIMES.map((regime) => (
              <div key={regime} className="panel" style={{ background: "var(--bg-inset)" }}>
                <div className="panel-title">{regime}</div>
                {MODELS.map((model) => (
                  <RangeField
                    key={model}
                    label={model}
                    value={form.regime_weights[regime][model] ?? 0}
                    min={0}
                    max={1}
                    step={0.05}
                    onChange={(v) => setRegimeWeight(regime, model, v)}
                  />
                ))}
              </div>
            ))}
          </div>
        </div>

        <div className="grid-2">
          <div className="panel" style={{ background: "var(--bg-inset)" }}>
            <div className="panel-title">Council — bounds & orthogonality</div>
            <RangeField label="weight_clip.min" value={form.weight_clip_min} min={0} max={0.5} step={0.05}
              onChange={(v) => set("weight_clip_min", v)} />
            <RangeField label="weight_clip.max" value={form.weight_clip_max} min={0.1} max={1} step={0.05}
              onChange={(v) => set("weight_clip_max", v)} />
            <RangeField label="IC rolling window" value={form.ic_rolling_window} min={20} max={252} step={5}
              onChange={(v) => set("ic_rolling_window", v)} />
            <RangeField label="Sharpe rolling window" value={form.sharpe_rolling_window} min={60} max={504} step={10}
              onChange={(v) => set("sharpe_rolling_window", v)} />
            <label className="row" style={{ marginTop: 10, textTransform: "none", letterSpacing: 0, fontSize: 13, fontWeight: 500, color: "var(--text)" }}>
              <input
                type="checkbox"
                checked={form.use_orthogonality}
                onChange={(e) => set("use_orthogonality", e.target.checked)}
              />
              Enable orthogonality monitor
            </label>
            <RangeField label="Max pairwise correlation" value={form.max_correlation} min={0.3} max={0.95} step={0.05}
              disabled={!form.use_orthogonality}
              onChange={(v) => set("max_correlation", v)} />
          </div>
          <div className="panel" style={{ background: "var(--bg-inset)" }}>
            <div className="panel-title">Portfolio constraints</div>
            <RangeField label="max_position" value={form.max_position} min={0.02} max={0.5} step={0.01}
              onChange={(v) => set("max_position", v)} />
            <RangeField label="max_turnover" value={form.max_turnover} min={0.05} max={1} step={0.05}
              onChange={(v) => set("max_turnover", v)} />
            <RangeField label="max_vol_ann" value={form.max_vol_ann} min={0.05} max={0.8} step={0.05}
              onChange={(v) => set("max_vol_ann", v)} />
            <RangeField label="sector_cap" value={form.sector_cap} min={0.1} max={1} step={0.05}
              onChange={(v) => set("sector_cap", v)} />
            <RangeField label="min_signal_strength" value={form.min_signal_strength} min={0} max={1} step={0.05}
              onChange={(v) => set("min_signal_strength", v)} />
          </div>
        </div>

        <label>
          Note (saved with the snapshot)
          <input value={form.note} onChange={(e) => set("note", e.target.value)} />
        </label>

        <div className="form-actions">
          <button className="btn btn-primary" onClick={onRun} disabled={submit.isPending || form.universe.length === 0}>
            {submit.isPending ? "Enqueuing…" : "Run Backtest"}
          </button>
          {submit.isSuccess && <p className="form-status">Submitted — the job appears in the table below once the worker picks it up.</p>}
        </div>
        {submit.error && <p className="form-error">{String(submit.error)}</p>}
      </div>

      <div>
        <h2>Jobs</h2>
        <DataTable
          rows={jobs.data?.jobs ?? ([] as JobEntry[])}
          columns={columns}
          emptyMessage="No backtest jobs yet."
          renderCell={(col, row: JobEntry) => {
            if (col === "state") return <StatusBadge state={row.state} />;
            if (col === "id" && (row.state === "running" || row.state === "queued")) {
              return (
                <span>
                  {row.id}{" "}
                  <button className="link-button" onClick={() => cancel.mutate(row.id)}>cancel</button>
                </span>
              );
            }
            return String(row[col as keyof JobEntry] ?? "");
          }}
        />
      </div>

      <div>
        <h2>Snapshots</h2>
        <SnapshotOverlay />
      </div>
    </div>
  );
}
