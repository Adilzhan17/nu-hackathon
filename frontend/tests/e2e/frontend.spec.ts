import { test, expect } from "@playwright/test";
import { readFileSync } from "node:fs";
const fixture = readFileSync("public/demo/cashier-preview.mp4");
test("upload validation, processing, playback, evidence and event seek", async ({ page }) => {
  await page.goto("/analysis/new");
  await expect(page.getByRole("button", { name: "Выбрать файл", exact: true })).toBeEnabled();
  await page.getByRole("button", { name: "Начать анализ" }).click();
  await expect(page.locator("#upload-error")).toContainText("Сначала выберите");
  const input = page.getByLabel("Выбрать видео .mov", { exact: true });
  await input.setInputFiles({ name: "wrong.mp4", mimeType: "video/mp4", buffer: fixture });
  await expect(page.locator("#upload-error")).toContainText("Нужен файл");
  await input.setInputFiles({ name: "cashier.MOV", mimeType: "video/quicktime", buffer: fixture });
  await expect(page.getByText("Файл готов к загрузке")).toBeVisible();
  await page.getByRole("button", { name: "Начать анализ" }).click();
  await expect(page.getByRole("heading", { name: "Анализируем видеозапись" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Результат анализа" })).toBeVisible({
    timeout: 15000,
  });
  await expect(page.getByText("Требует внимания", { exact: true })).toBeVisible();
  await page.locator(".event-item").nth(1).click();
  await expect
    .poll(() => page.locator("video").evaluate((v: HTMLVideoElement) => v.currentTime))
    .toBeCloseTo(58, 0);
  await page.getByRole("button", { name: "Воспроизвести", exact: true }).click();
  await expect
    .poll(() => page.locator("video").evaluate((v: HTMLVideoElement) => v.paused))
    .toBe(false);
  await page.getByRole("button", { name: "Пауза", exact: true }).click();
  await page.getByRole("button", { name: "Увеличить кадр 00:58" }).click();
  await expect(page.getByRole("dialog")).toBeVisible();
  await expect(
    page.getByRole("dialog").getByText(/Подтверждённый источник не указан/),
  ).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(page.getByRole("dialog")).not.toBeVisible();
  await page.reload();
  await expect(page.getByText("Исходная запись не открыта", { exact: true })).toBeVisible();
});
test("negative and failed uploaded scenarios", async ({ page }) => {
  for (const [radio, heading] of [
    ["Нет признаков", "Признаки не обнаружены"],
    ["Ошибка", "Анализ не завершён"],
  ]) {
    await page.goto("/analysis/new");
    await expect(page.getByRole("button", { name: "Выбрать файл", exact: true })).toBeEnabled();
    await page
      .getByLabel("Выбрать видео .mov", { exact: true })
      .setInputFiles({ name: "test.mov", mimeType: "video/quicktime", buffer: fixture });
    await page.getByRole("radio", { name: new RegExp(radio) }).check();
    await page.getByRole("button", { name: "Начать анализ" }).click();
    await expect(page.getByRole("heading", { name: heading, exact: true })).toBeVisible({
      timeout: 15000,
    });
  }
});
test("demo navigation, markers and fullscreen controls are available", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/");
  await page.getByRole("link", { name: "Посмотреть пример" }).click();
  await expect(page.getByRole("heading", { name: "Результат анализа" })).toBeVisible();
  await page.getByRole("button", { name: "Событие 2, 00:58", exact: true }).click();
  await expect
    .poll(() => page.locator("video").evaluate((v: HTMLVideoElement) => v.currentTime))
    .toBeCloseTo(58, 0);
  await expect(page.getByRole("button", { name: "Полноэкранный режим" })).toBeEnabled();
  await expect(page.locator(".evidence-thumbnail img")).toHaveJSProperty("naturalWidth", 960);
  expect(errors).toEqual([]);
});
test("missing jobs recover without hanging", async ({ page }) => {
  await page.goto("/analysis/nonexistent");
  await expect(page.getByRole("heading", { name: "Не удалось получить результат" })).toBeVisible({
    timeout: 12000,
  });
  await expect(page.getByRole("button", { name: "Запросить результат снова" })).toBeVisible();
});
test("mobile menu and all required widths avoid horizontal overflow", async ({ page }) => {
  for (const width of [1440, 1280, 1024, 768, 390]) {
    await page.setViewportSize({ width, height: 1000 });
    for (const route of [
      "/",
      "/analysis/new",
      "/analysis/demo-violation",
      "/analysis/demo-no_violation",
    ]) {
      await page.goto(route);
      await expect(page.locator("h1")).toBeVisible();
      expect(
        await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth),
      ).toBe(true);
    }
  }
  await page.getByRole("button", { name: "Открыть меню" }).click();
  await page.getByRole("navigation").getByRole("link", { name: "Новый анализ" }).click();
  await expect(page).toHaveURL(/\/analysis\/new$/);
  await expect(page.locator(".sidebar")).not.toHaveClass(/is-open/);
});
