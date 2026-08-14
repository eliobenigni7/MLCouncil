import { useQuery } from "@tanstack/react-query";
import { api } from "../api/client";

interface AlertItem {
  level?: string;
  title?: string;
  message?: string;
  severity?: string;
  model_name?: string;
  check_type?: string;
}

export function AlertBanner() {
  const { data } = useQuery({
    queryKey: ["alerts"],
    queryFn: () => api<AlertItem[]>("/api/monitoring/alerts"),
    refetchInterval: 120_000,
  });
  if (!data || data.length === 0) return null;
  return (
    <div className="alert-banner">
      {data.map((a, i) => {
        const level = (a.level ?? a.severity ?? "info").toLowerCase();
        const title = a.title ?? [a.model_name, a.check_type].filter(Boolean).join(" · ");
        return (
          <div key={i} className={`alert-item alert-${level}`}>
            <strong>{title}</strong> {a.message}
          </div>
        );
      })}
    </div>
  );
}
