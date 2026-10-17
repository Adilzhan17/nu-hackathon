import { test, expect } from "@playwright/test";
import { readFileSync } from "node:fs";
const fixture = readFileSync("public/demo/cashier-preview.mp4");
const evidence = readFileSync("public/demo/evidence.jpg").toString("base64");

test("live upload calls the API and displays its uncertain result", async ({ page }) => {
  let uploaded = false;
  await page.route("**/api/analysis", async (route) => {
    uploaded = true;
    expect(route.request().headers()["x-api-key"]).toBeUndefined();
    expect(route.request().headers()["content-type"]).toContain("multipart/form-data");
    await route.fulfill({ json: { job_id: "live-job" } });
  });
  await page.route("**/api/jobs/live-job", (route) =>
    route.fulfill({
      json: {
        job_id: "live-job",
        status: "done",
        verdict: "violation",
        certainty: "suspected",
        warnings: ["Типы нарушений не обучены"],
        violations: [
          {
            type: "Неклассифицированный признак возможного нарушения",
            timecode_sec: 24,
            description: "Проверьте фрагмент",
            certainty: "suspected",
            classification_status: "unclassified",
            regulatory_source: null,
            evidence_frames: [{ timestamp_sec: 24, image_base64: evidence }],
          },
        ],
      },
    }),
  );
  await page.goto("/analysis/new");
  await expect(page.getByRole("radio")).toHaveCount(0);
  await page
    .getByLabel("Выбрать видео .mov", { exact: true })
    .setInputFiles({ name: "test.mov", mimeType: "video/quicktime", buffer: fixture });
  await page.getByRole("button", { name: "Начать анализ" }).click();
  await expect(page.getByRole("heading", { name: "Результат анализа" })).toBeVisible();
  expect(uploaded).toBe(true);
  await expect(page.getByText("Типы нарушений не обучены", { exact: true })).toBeVisible();
  await expect(page.getByText("Тип нарушения не установлен", { exact: true })).toBeVisible();
  await expect(page.getByText("Подозрение — требует проверки", { exact: true })).toBeVisible();
  await expect(page.getByText(/Демонстрационный результат/)).toHaveCount(0);
});

test("live errors stay errors and never fall back to demonstration", async ({ page }) => {
  await page.route("**/api/analysis", (route) =>
    route.fulfill({ status: 401, json: { error: "Unauthorized" } }),
  );
  await page.goto("/analysis/new");
  await page
    .getByLabel("Выбрать видео .mov", { exact: true })
    .setInputFiles({ name: "test.mov", mimeType: "video/quicktime", buffer: fixture });
  await page.getByRole("button", { name: "Начать анализ" }).click();
  await expect(page.locator("#upload-error")).toContainText("API не авторизовал запрос");
  await expect(page).toHaveURL(/\/analysis\/new$/);
  await page.route("**/api/jobs/timeout-job", (route) =>
    route.fulfill({
      json: {
        job_id: "timeout-job",
        status: "failed",
        error: { code: "deadline_exceeded", message: "Истекло 300 секунд" },
      },
    }),
  );
  await page.goto("/analysis/timeout-job");
  await expect(page.getByRole("heading", { name: "Анализ не завершён" })).toBeVisible();
  await expect(page.locator(".state-panel [role=alert]")).toContainText("Истекло 300 секунд");
});

test("missing certainty is shown as unknown; an explicit demo remains labelled", async ({
  page,
}) => {
  await page.route("**/api/jobs/no-certainty", (route) =>
    route.fulfill({
      json: {
        job_id: "no-certainty",
        status: "done",
        verdict: "violation",
        violations: [
          {
            type: "Признак для проверки",
            timecode_sec: 24,
            description: "Проверьте фрагмент",
            evidence_frames: [{ timestamp_sec: 24, image_base64: evidence }],
          },
        ],
      },
    }),
  );
  await page.goto("/analysis/no-certainty");
  await expect(page.getByText("Достоверность не указана", { exact: true })).toBeVisible();
  await page.goto("/analysis/demo-violation");
  await expect(page.getByText("Учебный пример. События и кадры подготовлены заранее.", { exact: true })).toBeVisible();
});
