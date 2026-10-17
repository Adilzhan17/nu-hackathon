import { defineConfig } from "@playwright/test";
const demo = process.env.E2E_MODE === "demo";
const mockFlag = demo ? "true" : "false";
export default defineConfig({
  testDir: "./tests/e2e",
  testMatch: demo ? "frontend.spec.ts" : "live.spec.ts",
  timeout: 30_000,
  fullyParallel: true,
  workers: 3,
  reporter: "list",
  use: {
    baseURL: "http://127.0.0.1:3000",
    ...(process.env.PLAYWRIGHT_EXECUTABLE_PATH
      ? { launchOptions: { executablePath: process.env.PLAYWRIGHT_EXECUTABLE_PATH } }
      : { channel: "chrome" }),
    headless: true,
    screenshot: "only-on-failure",
  },
  webServer: {
    command: `NEXT_PUBLIC_USE_MOCK_API=${mockFlag} npm run build && NEXT_PUBLIC_USE_MOCK_API=${mockFlag} npm run start`,
    url: "http://127.0.0.1:3000",
    reuseExistingServer: false,
    timeout: 180_000,
  },
});
