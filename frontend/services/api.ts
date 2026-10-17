import { z } from "zod";
import { analysisJobSchema, type AnalysisJob, type MockScenario } from "@/types/analysis";
import { DEMO_JOBS, mockAnalyze, mockGetJob } from "./mock-api";
export const USE_MOCK_API = process.env.NEXT_PUBLIC_USE_MOCK_API === "true";
export class ApiError extends Error {
  constructor(
    message: string,
    public status?: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}
async function request(path: string, init: RequestInit, signal?: AbortSignal) {
  const timeout = AbortSignal.timeout(init.method === "POST" ? 120_000 : 20_000);
  try {
    const response = await fetch(path, {
      ...init,
      cache: "no-store",
      signal: signal ? AbortSignal.any([signal, timeout]) : timeout,
    });
    if (!response.ok) {
      const messages: Record<number, string> = {
        401: "API не авторизовал запрос. Проверьте серверный API_KEY.",
        403: "Доступ к API запрещён. Обратитесь к разработчику сервера.",
        404: "Задача не найдена. Запустите новый анализ.",
        413: "Сервер отклонил размер файла. Выберите видео меньшего размера.",
        415: "Сервер не поддерживает этот формат видео.",
        422: "Сервер не принял файл. Проверьте формат видеозаписи.",
        429: "Слишком много запросов. Повторите попытку чуть позже.",
        503: "Сервис анализа не настроен или временно недоступен.",
        502: "Сервис анализа недоступен. Проверьте соединение с сервером.",
        504: "Сервис анализа не ответил в отведённое время.",
      };
      throw new ApiError(
        messages[response.status] ?? "Сервис анализа вернул ошибку. Повторите попытку.",
        response.status,
      );
    }
    try {
      return await response.json();
    } catch {
      throw new ApiError("Сервер вернул пустой или некорректный ответ.");
    }
  } catch (error) {
    if (signal?.aborted) throw error;
    if (error instanceof ApiError) throw error;
    throw new ApiError(
      timeout.aborted
        ? "Сервер не ответил вовремя. Проверьте соединение и повторите попытку."
        : "Не удалось связаться с сервером. Проверьте подключение и повторите попытку.",
    );
  }
}
export async function analyzeVideo(
  file: File,
  scenario: MockScenario,
  signal?: AbortSignal,
): Promise<{ job_id: string }> {
  if (USE_MOCK_API) return mockAnalyze(file, scenario, signal);
  const form = new FormData();
  form.append("file", file);
  const parsed = z
    .object({ job_id: z.string().regex(/^[a-zA-Z0-9_-]{1,160}$/) })
    .safeParse(await request("/api/analysis", { method: "POST", body: form }, signal));
  if (!parsed.success) throw new ApiError("В ответе сервера отсутствует идентификатор задачи.");
  return parsed.data;
}
export async function getJob(id: string, signal?: AbortSignal): Promise<AnalysisJob> {
  if (DEMO_JOBS.some((demo) => demo.id === id) || (USE_MOCK_API && id.startsWith("mock-")))
    return mockGetJob(id, signal);
  const parsed = analysisJobSchema.safeParse(
    await request(`/api/jobs/${encodeURIComponent(id)}`, { method: "GET" }, signal),
  );
  if (!parsed.success || parsed.data.job_id !== id)
    throw new ApiError("Результат не соответствует API-контракту. Проверьте ответ бэкенда.");
  return parsed.data;
}
