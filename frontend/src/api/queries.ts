import { useQuery } from "@tanstack/react-query";
import { api } from "./client";

export interface Series {
  dates: string[];
  values: number[];
}

export interface AttributionRow {
  date: string;
  model_name: string;
  weight: number | null;
  ic_rolling_30d: number | null;
  sharpe_rolling_60d: number | null;
  pnl_contribution: number | null;
}

export interface RegimeHistoryRow {
  date: string;
  regime: string;
  prob_bull: number | null;
  prob_bear: number | null;
  prob_transition: number | null;
}

export interface SidebarMetrics {
  sharpe_ytd: number;
  max_dd: number;
  ic_30d: number;
  regime: string;
  regime_prob: number;
  sharpe_delta: number;
  dd_delta: number;
  ic_delta: number;
}

export interface PortfolioRow {
  ticker: string;
  weight_target: number | null;
  weight_current: number | null;
  signal: number | null;
}

export interface FillQualityRow {
  ticker: string;
  median_is_bps: number | null;
  fill_count: number | null;
  lookup_slippage_bps: number | null;
  kappa_calibrated_bps: number | null;
}

export interface CanaryFlag {
  name: string;
  env: string;
  value: string;
  config_enabled: boolean;
  reverted: boolean;
  pending_enabled: boolean | null;
  effective_enabled: boolean;
  floor: number;
  min_days: number;
}

export interface CanaryState {
  state_file: string;
  exists: boolean;
  reverted_features: Record<string, { reverted_at?: string; reason?: string }>;
  pending_apply: Record<string, { enabled: boolean; at?: string }>;
  history: Record<string, unknown>;
}

export function useEquity(mode = "Paper Trading", tag?: string) {
  const tagPart = tag ? `&tag=${encodeURIComponent(tag)}` : "";
  return useQuery({
    queryKey: ["equity", mode, tag],
    queryFn: () => api<Series>(`/api/analytics/equity?mode=${encodeURIComponent(mode)}${tagPart}`),
    staleTime: 60_000,
  });
}

export function useBenchmark(mode = "Paper Trading") {
  return useQuery({
    queryKey: ["benchmark", mode],
    queryFn: () => api<Series>(`/api/analytics/benchmark?mode=${encodeURIComponent(mode)}`),
    staleTime: 60_000,
  });
}

export function useDailyReturns(mode = "Paper Trading") {
  return useQuery({
    queryKey: ["returns", mode],
    queryFn: () => api<Series>(`/api/analytics/returns?mode=${encodeURIComponent(mode)}`),
    staleTime: 60_000,
  });
}

export function useAttribution(start?: string, end?: string) {
  return useQuery({
    queryKey: ["attribution", start, end],
    queryFn: () =>
      api<{ records: AttributionRow[] }>(
        `/api/analytics/attribution${start || end ? `?start=${start ?? ""}&end=${end ?? ""}` : ""}`,
      ),
    staleTime: 60_000,
  });
}

export function useIcHistory() {
  return useQuery({
    queryKey: ["ic-history"],
    queryFn: () =>
      api<{ records: { date: string; lgbm: number | null; sentiment: number | null; hmm: number | null }[] }>(
        "/api/analytics/ic-history",
      ),
    staleTime: 60_000,
  });
}

export function useWeightsHistory() {
  return useQuery({
    queryKey: ["weights-history"],
    queryFn: () =>
      api<{ records: { date: string; lgbm: number | null; sentiment: number | null; hmm: number | null }[] }>(
        "/api/analytics/weights-history",
      ),
    staleTime: 60_000,
  });
}

export function useCurrentRegime() {
  return useQuery({
    queryKey: ["regime-current"],
    queryFn: () =>
      api<{ regime: string; bull: number; bear: number; transition: number }>("/api/analytics/regime/current"),
    staleTime: 60_000,
  });
}

export function useRegimeHistory() {
  return useQuery({
    queryKey: ["regime-history"],
    queryFn: () => api<{ records: RegimeHistoryRow[] }>("/api/analytics/regime/history"),
    staleTime: 60_000,
  });
}

export function useSidebarMetrics() {
  return useQuery({
    queryKey: ["sidebar-metrics"],
    queryFn: () => api<SidebarMetrics>("/api/analytics/sidebar-metrics"),
    staleTime: 60_000,
  });
}

export function usePortfolioSnapshot() {
  return useQuery({
    queryKey: ["portfolio-snapshot"],
    queryFn: () => api<{ records: PortfolioRow[] }>("/api/analytics/portfolio-snapshot"),
    staleTime: 60_000,
  });
}

export function useOptimizationDiagnostics(asOf: string) {
  return useQuery({
    queryKey: ["optimization-diagnostics", asOf],
    queryFn: () => api<Record<string, unknown>>(`/api/analytics/optimization-diagnostics?as_of=${asOf}`),
    staleTime: 60_000,
  });
}

export function useWeightsLog(asOf: string) {
  return useQuery({
    queryKey: ["weights-log", asOf],
    queryFn: () => api<Record<string, unknown>>(`/api/analytics/weights-log?as_of=${asOf}`),
    staleTime: 60_000,
  });
}

export function useFillQuality() {
  return useQuery({
    queryKey: ["fill-quality"],
    queryFn: () => api<{ records: FillQualityRow[] }>("/api/analytics/fill-quality"),
    staleTime: 60_000,
  });
}

export function useCostCalibration() {
  return useQuery({
    queryKey: ["calibration"],
    queryFn: () => api<Record<string, unknown>>("/api/analytics/calibration"),
    staleTime: 60_000,
  });
}

export function useAlerts() {
  return useQuery({
    queryKey: ["alerts"],
    queryFn: () =>
      api<{ level: string; title: string; message?: string }[]>("/api/monitoring/alerts"),
    refetchInterval: 120_000,
  });
}

export function useCanaryFlags() {
  return useQuery({
    queryKey: ["canary-flags"],
    queryFn: () => api<{ features: CanaryFlag[] }>("/api/canary/flags"),
    staleTime: 30_000,
  });
}

export function useCanaryState() {
  return useQuery({
    queryKey: ["canary-state"],
    queryFn: () => api<CanaryState>("/api/canary/state"),
    staleTime: 30_000,
  });
}
