import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "../api/client";
import { StatusBadge } from "../components/StatusBadge";

interface OrderRecord {
  ticker?: string;
  symbol?: string;
  direction?: string;
  side?: string;
  quantity?: number;
  target_weight?: number;
  [key: string]: unknown;
}

export function PortfolioPage() {
  const [date, setDate] = useState<string | null>(null);

  const weights = useQuery({
    queryKey: ["portfolio-weights"],
    queryFn: async () => {
      const payload = await api<Record<string, unknown>>("/api/portfolio/weights");
      const inner = (payload as { weights?: Record<string, unknown> }).weights;
      return inner ?? payload;
    },
    retry: false,
  });

  const dates = useQuery({
    queryKey: ["portfolio-dates"],
    queryFn: () => api<string[]>("/api/portfolio/orders/dates"),
    retry: false,
  });

  const orders = useQuery({
    queryKey: ["portfolio-orders", date],
    queryFn: () => api<OrderRecord[]>(`/api/portfolio/orders/${date}`),
    enabled: date !== null,
    retry: false,
  });

  const weightEntries = Object.entries(weights.data ?? {});
  const availableDates = dates.data ?? [];

  return (
    <div className="page">
      <h1>Portfolio</h1>
      <p className="caption">
        Current target weights from the optimizer and per-date order files.
      </p>

      <div>
        <h2>Current weights</h2>
        {weights.isLoading ? (
          <div className="page-empty">Loading…</div>
        ) : weightEntries.length === 0 ? (
          <div className="page-empty">No weights yet — run the pipeline to produce portfolio weights.</div>
        ) : (
          <div className="table-wrap">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Ticker</th>
                  <th className="num">Weight</th>
                </tr>
              </thead>
              <tbody>
                {weightEntries.map(([ticker, weight]) => (
                  <tr key={ticker}>
                    <td className="mono">{ticker}</td>
                    <td className="num">{`${(Number(weight) * 100).toFixed(1)}%`}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <div>
        <h2>Order history</h2>
        <div className="row" style={{ marginBottom: 12 }}>
          <label>
            Order date
            <select
              value={date ?? ""}
              onChange={(e) => setDate(e.target.value || null)}
              disabled={availableDates.length === 0}
            >
              <option value="">{availableDates.length === 0 ? "No orders" : "Select date…"}</option>
              {availableDates.map((d) => (
                <option key={d} value={d}>
                  {d}
                </option>
              ))}
            </select>
          </label>
          {date && <span className="mono muted">{date}</span>}
        </div>
        {date === null ? (
          <div className="page-empty" style={{ padding: "24px 16px" }}>
            {availableDates.length === 0 ? "No order files found." : "Pick a date to view its orders."}
          </div>
        ) : orders.isLoading ? (
          <div className="page-empty">Loading…</div>
        ) : orders.error ? (
          <div className="page-empty">No orders for {date}.</div>
        ) : (
          <div className="table-wrap">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Ticker</th>
                  <th>Direction</th>
                  <th className="num">Quantity</th>
                  <th className="num">Target weight</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {(orders.data ?? []).map((o, i) => (
                  <tr key={i}>
                    <td className="mono">{o.ticker ?? o.symbol ?? "—"}</td>
                    <td>{(o.direction ?? o.side ?? "—").toUpperCase()}</td>
                    <td className="num">{o.quantity != null ? String(o.quantity) : "—"}</td>
                    <td className="num">
                      {o.target_weight != null ? `${(o.target_weight * 100).toFixed(1)}%` : "—"}
                    </td>
                    <td>{o.status ? <StatusBadge state={String(o.status)} /> : <span className="muted">—</span>}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
