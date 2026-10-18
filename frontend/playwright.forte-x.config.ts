import { defineConfig } from "@playwright/test";
// Existing API-contract UI tests against the isolated running production build.
// These use mocked API responses; the separate forte-x-production.cjs does not.
export default defineConfig({
  testDir: "./tests/e2e", testMatch: "live.spec.ts", timeout: 30_000,
  workers: 1, reporter: "list", use: { baseURL: "http://127.0.0.1:3001", headless: true },
});
