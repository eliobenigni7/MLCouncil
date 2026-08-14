import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { http, HttpResponse } from "msw";
import { server } from "../test/server";
import { CanaryPage } from "./CanaryPage";

it("shows flags and applies a pending change", async () => {
  server.use(
    http.get("/api/canary/flags", () =>
      HttpResponse.json({
        features: [{
          name: "online_learning", env: "MLCOUNCIL_ONLINE_LEARNING", value: "true",
          config_enabled: true, reverted: false, pending_enabled: null,
          effective_enabled: true, floor: 0, min_days: 5,
        }],
      })),
    http.get("/api/canary/state", () =>
      HttpResponse.json({ state_file: "x", exists: true, reverted_features: {}, pending_apply: {}, history: {} })),
    http.post("/api/canary/apply", () => HttpResponse.json({ pending_changes: [], flags: [] })),
  );
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={qc}>
      <CanaryPage />
    </QueryClientProvider>,
  );
  expect(await screen.findByText(/online_learning/)).toBeInTheDocument();
  await userEvent.click(await screen.findByRole("button", { name: /apply/i }));
});
