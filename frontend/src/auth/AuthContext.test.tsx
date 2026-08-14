import { render, screen, waitFor } from "@testing-library/react";
import { AuthProvider, useAuth } from "./AuthContext";

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
