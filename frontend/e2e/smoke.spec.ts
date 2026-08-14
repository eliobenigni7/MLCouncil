import { expect, test } from "@playwright/test";

const USER = process.env.MLCOUNCIL_ADMIN_USERNAME ?? "admin";
const PASS = process.env.MLCOUNCIL_ADMIN_PASSWORD ?? "change-me";

test.skip(!process.env.RUN_E2E, "E2E requires the API running (RUN_E2E=1)");

test("smoke: login, navigate all sections, run a backtest job", async ({ page }) => {
  await page.goto("/login");
  await page.getByLabel("Username").fill(USER);
  await page.getByLabel("Password").fill(PASS);
  await page.getByRole("button", { name: /sign in/i }).click();
  await expect(page).toHaveURL(/\/$/);

  const sections = ["Performance", "Attribution", "Regime", "Backtest", "Promotion", "Canary", "Pipeline", "Trading", "Portfolio", "Configuration", "Monitoring", "Fill Quality"];
  for (const s of sections) {
    await page.getByRole("link", { name: new RegExp(s, "i") }).first().click();
    await expect(page.locator("h1").first()).toContainText(s, { timeout: 10_000 });
  }

  await page.getByRole("link", { name: /backtest/i }).click();
  // short range: the subprocess backtest job finishes quickly
  await page.getByLabel("Start").fill("2024-01-01");
  await page.getByLabel("End").fill("2024-02-01");
  await page.getByRole("button", { name: /run backtest/i }).click();
  await expect(page.getByText(/job-/)).toBeVisible({ timeout: 60_000 });
});
