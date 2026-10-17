import type { AnalysisJob, MockScenario, Violation } from "@/types/analysis";
import { getMetadata } from "@/lib/session";
export const DEMO_JOBS = [
  {
    id: "demo-violation",
    name: "cashier_014.mov",
    scenario: "violation" as const,
    duration: 90,
    label: "Возможные нарушения",
    note: "2 события для проверки",
  },
  {
    id: "demo-no_violation",
    name: "cashier_027.mov",
    scenario: "no_violation" as const,
    duration: 90,
    label: "Признаки не обнаружены",
    note: "Нет отмеченных событий",
  },
  {
    id: "demo-failed",
    name: "cashier_032.mov",
    scenario: "failed" as const,
    duration: 90,
    label: "Ошибка обработки",
    note: "Пример восстановления",
  },
];
function delay(ms: number, signal?: AbortSignal) {
  return new Promise<void>((resolve, reject) => {
    if (signal?.aborted) {
      reject(signal.reason);
      return;
    }
    const abort = () => {
      clearTimeout(timer);
      reject(signal?.reason);
    };
    const timer = setTimeout(() => {
      signal?.removeEventListener("abort", abort);
      resolve();
    }, ms);
    signal?.addEventListener("abort", abort, { once: true });
  });
}
export async function mockAnalyze(_file: File, _scenario: MockScenario, signal?: AbortSignal) {
  await delay(700, signal);
  return { job_id: `mock-${crypto.randomUUID()}` };
}
async function frameBase64(second: number, signal?: AbortSignal) {
  const response = await fetch(second === 58 ? "/demo/evidence-58.jpg" : "/demo/evidence.jpg", {
    signal,
  });
  if (!response.ok) throw new Error("Не удалось загрузить демонстрационный кадр.");
  const bytes = new Uint8Array(await response.arrayBuffer());
  let binary = "";
  bytes.forEach((byte) => (binary += String.fromCharCode(byte)));
  return btoa(binary);
}
export async function mockGetJob(id: string, signal?: AbortSignal): Promise<AnalysisJob> {
  await delay(300, signal);
  const preset = DEMO_JOBS.find((demo) => demo.id === id);
  const meta = getMetadata(id);
  if (!preset && !meta)
    throw new Error(
      "Задача не найдена в этой вкладке. Запустите новый анализ или откройте демопример.",
    );
  if (!preset && meta && Date.now() - meta.createdAt < 6500)
    return { job_id: id, status: "processing" };
  const scenario = preset?.scenario ?? meta?.scenario ?? "violation";
  if (scenario === "failed")
    return {
      job_id: id,
      status: "failed",
      error:
        "Демонстрационный сценарий: не удалось декодировать видеозапись. Выберите другой файл и повторите анализ.",
    };
  if (scenario === "no_violation")
    return {
      job_id: id,
      status: "done",
      verdict: "no_violation",
      violations: [],
      ...(preset ? { preview_url: "/demo/cashier-preview.mp4" } : {}),
    };
  const [frame, secondFrame] = await Promise.all([
    frameBase64(24, signal),
    frameBase64(58, signal),
  ]);
  const violations: Violation[] = [
    {
      type: "Передача наличных без видимой проверки документа",
      timecode_sec: 24,
      description:
        "В учебном сценарии отмечена передача наличных посетителю. Предъявление документа в предшествующем фрагменте не видно. Для вывода необходимы контекст операции и проверка специалистом.",
      evidence_frames: [{ timestamp_sec: 24, image_base64: frame }],
    },
    {
      type: "Наличные вне зоны наблюдения",
      timecode_sec: 58,
      description:
        "В учебном сценарии наличные перемещаются за границу наблюдаемой зоны. По одной записи нельзя установить назначение действия. Проверьте фрагмент и сопутствующие документы.",
      evidence_frames: [{ timestamp_sec: 58, image_base64: secondFrame }],
    },
  ];
  return {
    job_id: id,
    status: "done",
    verdict: "violation",
    violations,
    ...(preset ? { preview_url: "/demo/cashier-preview.mp4" } : {}),
  };
}
