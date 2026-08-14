import { useQuery } from "@tanstack/react-query";
import { api } from "../api/client";

interface AlertItem {
  level: string;
  title: string;
  message?: string;
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
      {data.map((a, i) => (
        <div key={i} className={`alert-item alert-${a.level}`}>
          <strong>{a.title}</strong> {a.message}
        </div>
      ))}
    </div>
  );
}
