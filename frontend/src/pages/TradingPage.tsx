import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../api/client";
import { KpiCard } from "../components/KpiCard";
import { ConfirmDialog } from "../components/ConfirmDialog";
import { StatusBadge } from "../components/StatusBadge";

interface Position {
  symbol?: string;
  qty?: number | string;
  avg_price?: number | string;
  current_price?: number | string;
  unrealized_pl?: number | string;
  unrealized_pl_pc?: number | string;
}

interface PendingOrder {
  ticker?: string;
  direction?: string;
  target_weight?: number;
  notional?: number | string;
  quantity?: number | string;
}

interface Trade {
  symbol?: string;
  side?: string;
  qty?: number | string;
  status?: string;
  submitted_at?: string;
}

interface ExecuteResult {
  date?: string;
  orders_submitted?: number;
  orders_rejected?: number;
  liquidations?: number;
  error?: string | null;
}

function money(value: unknown): string {
  const n = Number(value ?? 0);
  return `$${n.toLocaleString(undefined, { maximumFractionDigits: 0 })}`;
}

export function TradingPage() {
  const qc = useQueryClient();
  const [confirmAction, setConfirmAction] = useState<"execute" | "liquidate" | null>(null);

  const status = useQuery({
    queryKey: ["trading-status"],
    queryFn: () =>
      api<{
        connected?: boolean;
        paper?: boolean;
        runtime_profile?: string;
        paused?: boolean;
        kill_switch_active?: boolean;
        account?: { equity?: number; buying_power?: number; portfolio_value?: number };
        positions?: Position[];
        error?: string | null;
      }>("/api/trading/status"),
    refetchInterval: 30_000,
  });

  const latest = useQuery({
    queryKey: ["trading-latest"],
    queryFn: () => api<{ date: string }>("/api/trading/orders/latest"),
    retry: false,
  });
  const orderDate = latest.data?.date ?? "";

  const pending = useQuery({
    queryKey: ["trading-pending", orderDate],
    queryFn: () => api<{ date: string; orders: PendingOrder[] }>(`/api/trading/orders/pending/${orderDate}`),
    enabled: orderDate.length > 0,
    retry: false,
  });

  const history = useQuery({
    queryKey: ["trading-history"],
    queryFn: () => api<{ trades: Trade[] }>("/api/trading/history?days=7"),
    retry: false,
  });

  const execute = useMutation({
    mutationFn: (date: string) =>
      api<ExecuteResult>("/api/trading/execute", { method: "POST", body: JSON.stringify({ date }) }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["trading-status"] });
      qc.invalidateQueries({ queryKey: ["trading-pending"] });
      qc.invalidateQueries({ queryKey: ["trading-history"] });
    },
  });

  const liquidate = useMutation({
    mutationFn: () => api<{ error?: string }>("/api/trading/liquidate", { method: "POST" }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["trading-status"] });
      qc.invalidateQueries({ queryKey: ["trading-history"] });
    },
  });

  const st = status.data;
  const disconnected = st?.connected === false;
  const paused = Boolean(st?.paused || st?.kill_switch_active);
  const busy = execute.isPending || liquidate.isPending;

  const executeBlocked = disconnected || paused;

  const trades = history.data?.trades ?? [];
  const pendingOrders = pending.data?.orders ?? [];

  return (
    <div className="page">
      <h1>Trading</h1>
      <p className="caption">
        Paper-trading execution console: account status, pending order book and execution history.
      </p>

      {status.isLoading ? (
        <div className="page-empty">Loading…</div>
      ) : (
        <>
          {disconnected ? (
            <div className="alert-item alert-error">
              <strong>Disconnected</strong> {st?.error ?? "Check Alpaca configuration."}
            </div>
          ) : (
            <div className="kpi-row">
              <KpiCard
                label="Mode"
                value={st?.paper ? "Paper" : "Live"}
                delta={{ value: st?.runtime_profile ?? "local", direction: "flat" }}
              />
              <KpiCard
                label="Buying power"
                value={money(st?.account?.buying_power)}
                delta={{ value: "available", direction: "flat" }}
              />
              <KpiCard label="Portfolio value" value={money(st?.account?.portfolio_value ?? st?.account?.equity)} />
              <KpiCard
                label="Automation"
                value={paused ? "Paused" : "Active"}
                tone={paused ? "warning" : "ok"}
                delta={{ value: orderDate ? `orders for ${orderDate}` : "no order files", direction: "flat" }}
              />
            </div>
          )}
        </>
      )}

      <div>
        <h2>Pending orders</h2>
        {pendingOrders.length === 0 ? (
          <div className="page-empty" style={{ padding: "24px 16px" }}>
            {orderDate ? "No pending orders for this date." : "No order files yet — run the daily pipeline first."}
          </div>
        ) : (
          <div className="table-wrap">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Ticker</th>
                  <th>Direction</th>
                  <th>Target weight</th>
                  <th>Notional</th>
                </tr>
              </thead>
              <tbody>
                {pendingOrders.map((o, i) => (
                  <tr key={i}>
                    <td className="mono">{o.ticker ?? "—"}</td>
                    <td>{(o.direction ?? "buy").toUpperCase()}</td>
                    <td className="num">{o.target_weight != null ? `${(o.target_weight * 100).toFixed(1)}%` : "—"}</td>
                    <td className="num">{money(o.notional ?? o.quantity ?? 0)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        <div className="row" style={{ marginTop: 12 }}>
          <button className="btn btn-primary" onClick={() => setConfirmAction("execute")} disabled={executeBlocked || busy}>
            {execute.isPending ? "Executing…" : "Execute Orders"}
          </button>
          <button className="btn btn-danger" onClick={() => setConfirmAction("liquidate")} disabled={busy || !st?.connected}>
            Liquidate All
          </button>
          {executeBlocked && <span className="muted">{disconnected ? "Trading connection unavailable." : "Trading is paused."}</span>}
        </div>
        {execute.data && (
          <div className="panel" style={{ marginTop: 12, background: "var(--bg-inset)" }}>
            <div className="panel-title">Execution result</div>
            <div className="row">
              <span className="tag">submitted: {execute.data.orders_submitted ?? 0}</span>
              <span className="tag">rejected: {execute.data.orders_rejected ?? 0}</span>
              <span className="tag">liquidations: {execute.data.liquidations ?? 0}</span>
            </div>
          </div>
        )}
        {execute.error && <p className="form-error">{String(execute.error)}</p>}
        {liquidate.error && <p className="form-error">{String(liquidate.error)}</p>}
      </div>

      <div>
        <h2>Positions</h2>
        {!st || (st.positions?.length ?? 0) === 0 ? (
          <div className="page-empty" style={{ padding: "24px 16px" }}>No open positions.</div>
        ) : (
          <div className="table-wrap">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Symbol</th>
                  <th className="num">Qty</th>
                  <th className="num">Avg price</th>
                  <th className="num">Current</th>
                  <th className="num">Unrealized P&L</th>
                </tr>
              </thead>
              <tbody>
                {st.positions!.map((p, i) => {
                  const pl = Number(p.unrealized_pl ?? 0);
                  return (
                    <tr key={i}>
                      <td className="mono">{p.symbol}</td>
                      <td className="num">{String(p.qty ?? "")}</td>
                      <td className="num">{money(p.avg_price)}</td>
                      <td className="num">{money(p.current_price)}</td>
                      <td className="num" style={{ color: pl >= 0 ? "var(--success)" : "var(--danger)" }}>
                        {pl >= 0 ? "+" : ""}{pl.toFixed(2)} ({Number(p.unrealized_pl_pc ?? 0).toFixed(1)}%)
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <div>
        <h2>Trade history (7d)</h2>
        {trades.length === 0 ? (
          <div className="page-empty" style={{ padding: "24px 16px" }}>No trade history.</div>
        ) : (
          <div className="table-wrap">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Symbol</th>
                  <th>Side</th>
                  <th className="num">Qty</th>
                  <th>Status</th>
                  <th>Submitted</th>
                </tr>
              </thead>
              <tbody>
                {trades.map((t, i) => (
                  <tr key={i}>
                    <td className="mono">{t.symbol ?? "—"}</td>
                    <td>{t.side ?? "—"}</td>
                    <td className="num">{String(t.qty ?? "—")}</td>
                    <td><StatusBadge state={t.status ?? "unknown"} /></td>
                    <td className="muted">{t.submitted_at ?? "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <ConfirmDialog
        open={confirmAction !== null}
        title={confirmAction === "execute" ? "Execute pending orders?" : "Liquidate all positions?"}
        body={
          confirmAction === "execute"
            ? orderDate
              ? `Submit the pending orders for ${orderDate} to the paper broker.`
              : "Submit the pending orders for the latest available order file to the paper broker."
            : "Sell every open position at market. This cannot be undone."
        }
        confirmLabel="Confirm"
        onCancel={() => setConfirmAction(null)}
        onConfirm={() => {
          if (confirmAction === "execute") execute.mutate(orderDate);
          if (confirmAction === "liquidate") liquidate.mutate();
          setConfirmAction(null);
        }}
      />
    </div>
  );
}
