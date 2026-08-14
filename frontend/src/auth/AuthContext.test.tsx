import { render, screen, waitFor } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { AuthProvider, useAuth } from "./AuthContext";
import { server } from "../test/server";

function Probe() {
  const { status, username } = useAuth();
  return <div data-testid="probe">{status}:{username ?? "none"}</div>;
}

it("loads session on mount", async () => {
  render(
    <AuthProvider>
      <Probe />
    </AuthProvider>,
  );
  await waitFor(() => expect(screen.getByTestId("probe")).toHaveTextContent("authenticated:admin"));
});

it("does not redirect when unauthenticated on the login page", async () => {
  server.use(http.get("/api/auth/me", () => HttpResponse.json({}, { status: 401 })));
  const originalHref = window.location.href;
  const originalAssign = window.location.assign;
  const assignSpy = vi.fn();
  // la pagina di login: /login
  window.history.pushState({}, "", "/login");
  Object.defineProperty(window, "location", { value: { ...window.location, assign: assignSpy } });
  try {
    render(
      <AuthProvider>
        <Probe />
      </AuthProvider>,
    );
    await waitFor(() => expect(screen.getByTestId("probe")).toHaveTextContent("unauthenticated:none"));
    expect(assignSpy).not.toHaveBeenCalled();
  } finally {
    window.history.pushState({}, "", originalHref);
    Object.defineProperty(window, "location", { value: { ...window.location, assign: originalAssign } });
  }
});
