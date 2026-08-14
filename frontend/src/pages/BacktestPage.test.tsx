import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { http, HttpResponse } from "msw";
import { server } from "../test/server";
import { BacktestPage } from "./BacktestPage";

vi.mock("react-plotly.js", () => ({
  default: (props: { data: unknown; layout: { title?: { text?: string } } }) => (
    <div data-testid="plot">{props.layout?.title?.text ?? ""}</div>
  ),
}));

it("submits a backtest job and shows the job id", async () => {
  server.use(
    http.post("/api/experiments/backtest", () => HttpResponse.json({ job_id: "job-abc", status: "queued" })),
    http.get("/api/experiments/jobs", () =>
      HttpResponse.json({ jobs: [{ id: "job-abc", state: "running", params: { note: "test" }, created_at: "2026-08-14T00:00:00Z" }] }),
    ),
    http.get("/api/experiments/snapshots", () => HttpResponse.json({ snapshots: [] })),
  );
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={qc}>
      <BacktestPage />
    </QueryClientProvider>,
  );
  await userEvent.click(await screen.findByRole("button", { name: /run backtest/i }));
  expect(await screen.findByText(/job-abc/)).toBeInTheDocument();
});
