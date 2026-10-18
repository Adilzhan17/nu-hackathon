import { readFile } from "node:fs/promises";
import path from "node:path";
import Link from "next/link";

export const dynamic = "force-dynamic";
export const metadata = { title: "Тесты Forte X" };
const root = process.env.FORTE_PROJECT_ROOT || "/root/nuhack";
async function report(file: string) {
  try { return JSON.parse(await readFile(path.join(root, file), "utf8")); }
  catch { return null; }
}
export default async function TestsPage() {
  const [evaluation, receipt, browser] = await Promise.all([
    report("runs/context_pipeline/detector_evaluation_v3/report.json"),
    report("runs/context_pipeline/site_delivery_receipt.json"),
    report("runs/forte_x/production_browser_test.json"),
  ]);
  const job = browser?.job_id || "788fb3b1b8a0439387d0c51ee17fd565";
  return <div className="page">
    <div className="page-heading"><div><h1>Тесты Forte X</h1><p>Фактические запуски и измерения. Демонстрационные ответы не используются.</p></div>
      <Link className="text-link" href="/analysis/new">Запустить анализ →</Link></div>
    <section className="panel forte-x-test-panel">
      <h2>Сквозной запуск через сайт</h2>
      <p>{browser ? `Статус: ${browser.status}. Проверка: ${browser.checked_at}.` : "Предыдущий полный запуск выполнен; проверка новой публикации ещё идёт."}</p>
      <p>Видео 15,6 с: предыдущий полный проход занял {receipt?.pipeline_wall_seconds ?? "—"} с. Детектор → наблюдения Qwen → память → общие правила.</p>
      <p><Link className="text-link" href={`/analysis/new?job=${job}`}>Открыть реальное видео, рамки, временные интервалы и доказательства →</Link></p>
      <p>Нарушение не подтверждено: недостаточно данных. Это проверка интеграции на известном train-видео, а не независимый тест качества.</p>
      <p>Проверки кода предыдущей поставки: {receipt?.tests_passed ?? "нет отчёта"}. {browser?.assertions?.join(" · ")}</p>
    </section>
    <section className="panel forte-x-test-panel">
      <h2>Деньги и документы: старый и новый checkpoint</h2>
      <p>14 реальных кадров отдельного исходного видео: 9 рамок денег, 12 документов/ID. Порог 0,25; IoU 0,5. Вырезки исходного кадра 384 px с перекрытием и NMS.</p>
      <div style={{overflowX:"auto"}}><table className="forte-x-metrics"><thead><tr><th>Checkpoint / объект</th><th>Precision</th><th>Recall</th><th>AP50</th><th>mAP50–95</th></tr></thead><tbody>
        {["old_tiles","new_tiles"].flatMap(condition => ["cash","document_id"].map(cls => {
          const m = evaluation?.conditions?.[condition]?.per_class?.[cls];
          const percent = (v: number | undefined) => v === undefined ? "—" : `${(v*100).toFixed(1)}%`;
          return <tr key={condition+cls}><td>{condition === "old_tiles" ? "Старый, активный" : "Новый, отклонён"} / {cls === "cash" ? "деньги" : "документ / ID"}</td>
            <td>{m?.precision_defined === false ? "— нет рамок" : percent(m?.precision)}</td><td>{percent(m?.recall)}</td><td>{percent(m?.ap50)}</td><td>{percent(m?.map50_95)}</td></tr>;
        }))}</tbody></table></div>
      <p>Новое дообучение не прошло проверку: деньги пропущены, точность документов ухудшилась. На сайте сохранён предыдущий дообученный детектор. Это малая диагностика на одном исходнике, не независимый test.</p>
      <p>Для метрик «в руках / на столе» нет подтверждённых тегов. Синтетика использовалась только в train.</p>
      <a className="text-link" href="/scenarios">Кадры пропусков, версии моделей, очередь проверки и CPU/RAM/GPU →</a>
    </section>
    <section className="panel forte-x-test-panel"><h2>Что остаётся ограничением</h2>
      <p>Forte X доступен на основном сайте. Автоматические рамки, действия и выводы требуют проверки человеком. Подлинность банкнот и личности не определяются.</p>
      <p>Пока до 60 секунд для полного контекстного прохода. Общий лимит запроса — 5 минут. Занятый GPU возвращает явную ошибку.</p>
      <p>VideoMAE использует экспериментальные NC-веса; YOLO — AGPL/Enterprise. Публикация интерфейса не меняет лицензионные условия моделей.</p>
    </section>
  </div>;
}
