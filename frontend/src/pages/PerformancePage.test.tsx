import { render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { http, HttpResponse } from "msw";
import { server } from "../test/server";
import { PerformancePage } from "./PerformancePage";

vi.mock("react-plotly.js", () => ({
  default: (props: { data: unknown; layout: { title?: { text?: string } } }) => (
    <div data-testid="plot">{props.layout?.title?.text ?? ""}</div>
  ),
}));

it("renders equity chart title after data loads", async () => {
  server.use(
    http.get("/api/analytics/equity", () => HttpResponse.json({ dates: ["2024-01-01"], values: [100] })),
    http.get("/api/analytics/benchmark", () => HttpResponse.json({ dates: ["2024-01-01"], values: [100] })),
    http.get("/api/analytics/returns", () => HttpResponse.json({ dates: ["2024-01-01"], values: [0.01] })),
  );
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={qc}>
      <PerformancePage />
    </QueryClientProvider>,
  );
  expect(await screen.findByText(/Equity & benchmark/)).toBeInTheDocument();
});
