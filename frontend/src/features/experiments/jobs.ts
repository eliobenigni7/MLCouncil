import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../../api/client";

export interface JobEntry {
  id: string;
  state: "queued" | "running" | "succeeded" | "cancelled" | "failed";
  params: Record<string, unknown>;
  created_at: string;
  snapshot_path?: string | null;
  error?: string;
}

export interface BacktestForm {
  start_date: string;
  end_date: string;
  universe: string[];
  initial_capital: number;
  slippage_bps: number;
  commission_bps: number;
  regime_weights: Record<string, Record<string, number>>;
  weight_clip_min: number;
  weight_clip_max: number;
  ic_rolling_window: number;
  sharpe_rolling_window: number;
  use_orthogonality: boolean;
  max_correlation: number;
  max_position: number;
  max_turnover: number;
  max_vol_ann: number;
  sector_cap: number;
  min_signal_strength: number;
  note: string;
}

export function useSubmitBacktest() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (params: BacktestForm) =>
      api<{ job_id: string; status: string }>("/api/experiments/backtest", {
        method: "POST",
        body: JSON.stringify({ params }),
      }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["jobs"] }),
  });
}

export function useJobs(pollMs = 5000) {
  return useQuery({
    queryKey: ["jobs"],
    queryFn: () => api<{ jobs: JobEntry[] }>("/api/experiments/jobs"),
    refetchInterval: (query) => {
      const jobs = query.state.data?.jobs ?? [];
      return jobs.some((j) => j.state === "running" || j.state === "queued") ? 2000 : pollMs;
    },
  });
}

export function useCancelJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (jobId: string) => api<JobEntry>(`/api/experiments/jobs/${jobId}/cancel`, { method: "POST" }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["jobs"] }),
  });
}
