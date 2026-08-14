import { http, HttpResponse } from "msw";
import { setupServer } from "msw/node";

export const server = setupServer(
  http.get("/api/auth/me", () => HttpResponse.json({ authenticated: true, username: "admin" })),
);
