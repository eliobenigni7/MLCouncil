export class ApiError extends Error {
  constructor(public status: number, public code: string, public message: string, public detail: string) {
    super(message);
  }
}

function csrfToken(): string | null {
  const match = document.cookie.match(/(?:^|;\s*)mlcouncil_csrf=([^;]+)/);
  return match ? decodeURIComponent(match[1]) : null;
}

export async function api<T>(path: string, options: RequestInit = {}): Promise<T> {
  const headers: Record<string, string> = { ...(options.headers as Record<string, string>) };
  const method = (options.method ?? "GET").toUpperCase();
  if (method !== "GET" && method !== "HEAD") {
    const token = csrfToken();
    if (token) headers["X-CSRF-Token"] = token;
  }
  if (options.body) headers["Content-Type"] = "application/json";
  const resp = await fetch(path, { ...options, headers, credentials: "same-origin" });
  if (resp.status === 401) {
    // Evita il loop: sulla pagina di login il probe /api/auth/me fallisce
    // volutamente con 401 — niente redirect, è già il punto di ingresso.
    if (!window.location.pathname.startsWith("/login")) {
      window.location.href = "/login";
    }
    throw new ApiError(401, "not_authenticated", "Not logged in", "");
  }
  let body: unknown = null;
  try {
    body = await resp.json();
  } catch {
    /* non-JSON body */
  }
  if (!resp.ok) {
    const err = (body as { error?: { code?: string; message?: string; detail?: string } })?.error;
    throw new ApiError(resp.status, err?.code ?? "http_error", err?.message ?? resp.statusText, err?.detail ?? "");
  }
  return body as T;
}

export const authApi = {
  login: (username: string, password: string) =>
    api<{ authenticated: boolean; username: string }>("/api/auth/login", {
      method: "POST",
      body: JSON.stringify({ username, password }),
    }),
  logout: () => api<{ authenticated: boolean }>("/api/auth/logout", { method: "POST" }),
  me: () => api<{ authenticated: boolean; username: string }>("/api/auth/me"),
};
