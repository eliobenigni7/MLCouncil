import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { http, HttpResponse } from "msw";
import { server } from "../test/server";
import { TradingPage } from "./TradingPage";

it("requires confirmation before execute", async () => {
  server.use(
    http.get("/api/trading/status", () => HttpResponse.json({ account: { equity: 100000, buying_power: 50000 }, positions: [], pending_orders: [] })),
    http.get("/api/trading/orders/latest", () => HttpResponse.json({ orders: [] })),
    http.get("/api/trading/history", () => HttpResponse.json({ trades: [] })),
    http.post("/api/trading/execute", () => HttpResponse.json({ ok: true })),
  );
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={qc}>
      <TradingPage />
    </QueryClientProvider>,
  );
  await userEvent.click(await screen.findByRole("button", { name: /execute/i }));
  expect(await screen.findByRole("button", { name: /confirm/i })).toBeInTheDocument();
});
