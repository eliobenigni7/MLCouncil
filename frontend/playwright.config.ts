import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e",
  timeout: 60_000,
  use: {
    baseURL: process.env.UI_BASE_URL ?? "http://localhost:8000",
    ...devices["Desktop Chrome"],
  },
  // Il test richiede l'API in esecuzione; in CI viene skippato
  grep: process.env.RUN_E2E ? undefined : /smoke/,
});
