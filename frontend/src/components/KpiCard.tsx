export function KpiCard({
  label,
  value,
  delta,
  tone,
}: {
  label: string;
  value: string;
  delta?: { value: string; direction: "up" | "down" | "flat" };
  tone?: "ok" | "warning" | "error";
}) {
  return (
    <div className="kpi-card">
      <div className="kpi-label">{label}</div>
      <div className={`kpi-value${tone ? ` ${tone}` : ""}`}>{value}</div>
      {delta && (
        <div className={`kpi-delta ${delta.direction === "up" ? "up" : delta.direction === "down" ? "down" : ""}`}>
          {delta.direction === "up" ? "▲" : delta.direction === "down" ? "▼" : "•"} {delta.value}
        </div>
      )}
    </div>
  );
}
