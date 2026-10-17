"use client";
import { useEffect, useState } from "react";
import Link from "next/link";
import {
  ArrowLeft,
  CircleAlert,
  CircleCheck,
  Clock3,
  FileVideo2,
  Info,
  LoaderCircle,
  Plus,
  RotateCcw,
  ScanLine,
  ShieldCheck,
  TriangleAlert,
} from "lucide-react";
import { useAnalysis } from "@/hooks/use-analysis";
import { Button } from "@/components/ui/button";
import { VideoPlayer } from "@/components/video/video-player";
import { EvidencePanel } from "@/components/violations/evidence-panel";
import { getMetadata, getVideoUrl } from "@/lib/session";
import { DEMO_JOBS } from "@/services/mock-api";
import { cn, timecode } from "@/lib/utils";
import type { UploadMetadata } from "@/types/analysis";
export function AnalysisView({ id }: { id: string }) {
  const { job, error, retrying, retry } = useAnalysis(id);
  const [meta, setMeta] = useState<UploadMetadata>();
  const [source, setSource] = useState<string>();
  const [selected, setSelected] = useState(0);
  const [seek, setSeek] = useState<{ seconds: number; nonce: number }>();
  useEffect(() => {
    setMeta(getMetadata(id));
    setSource(getVideoUrl(id));
  }, [id]);
  const preset = DEMO_JOBS.find((d) => d.id === id);
  const demo = !!preset || !!meta?.mock;
  const filename = meta?.name ?? preset?.name ?? "Видеозапись";
  const events = job?.status === "done" ? job.violations : [];
  function goTo(seconds: number) {
    setSeek((old) => ({ seconds, nonce: (old?.nonce ?? 0) + 1 }));
  }
  function selectEvent(index: number) {
    setSelected(index);
    goTo(events[index].timecode_sec);
  }
  if (error || job?.status === "failed")
    return (
      <div className="page">
        <Link className="back-link" href="/analysis">
          <ArrowLeft size={16} />К результатам
        </Link>
        <div className="panel state-panel">
          <span className="state-icon failed">
            <CircleAlert size={32} />
          </span>
          {demo && <span className="badge badge-neutral">Демосценарий</span>}
          <h1>{error ? "Не удалось получить результат" : "Анализ не завершён"}</h1>
          <p className="state-filename">{filename}</p>
          <p role="alert">
            {error ??
              (job?.status === "failed"
                ? (typeof job.error === "string" ? job.error : job.error?.message) ||
                  "Обработка завершилась ошибкой. Выберите другую запись и повторите анализ."
                : "")}
          </p>
          <div className="state-actions">
            {error && (
              <Button onClick={retry}>
                <RotateCcw size={17} />
                Запросить результат снова
              </Button>
            )}
            <Button asChild variant={error ? "outline" : "default"}>
              <Link href="/analysis/new">
                <Plus size={17} />
                Новый анализ
              </Link>
            </Button>
          </div>
          <details className="disclosure job-details">
            <summary>Данные задачи</summary>
            <code>{id}</code>
          </details>
        </div>
      </div>
    );
  if (!job || job.status === "processing")
    return (
      <div className="page">
        <Link className="back-link" href="/analysis">
          <ArrowLeft size={16} />К результатам
        </Link>
        <div className="panel state-panel processing-panel">
          <div className="processing-visual">
            <ScanLine size={52} strokeWidth={1.2} />
            <span />
          </div>
          <div className="badge badge-neutral">
            <LoaderCircle className="spin" size={13} />
            {!job ? "Получение статуса" : "Обработка"}
          </div>
          <h1>{!job ? "Открываем задачу" : "Анализируем видеозапись"}</h1>
          <p className="state-filename">
            <FileVideo2 size={17} />
            {filename}
          </p>
          <p>
            {demo ? "Демосценарий. Модель не запускается." : "Это может занять несколько минут."}
          </p>
          <div className="indeterminate-track">
            <span />
          </div>
          <p className="processing-status" role="status">
            {retrying
              ? "Соединение прервалось. Повторяем запрос статуса…"
              : "Статус обновляется автоматически."}
          </p>
          <details className="disclosure job-details">
            <summary>Данные задачи</summary>
            <code>{id}</code>
          </details>
        </div>
      </div>
    );
  const violation = job.verdict === "violation";
  const selectedEvent = events[selected] ?? events[0];
  return (
    <div className="page results-page">
      <Link className="back-link" href="/analysis">
        <ArrowLeft size={16} />К результатам
      </Link>
      <div className="page-heading results-heading">
        <div>
          <div className="inline-heading">
            <h1>Результат анализа</h1>
          </div>
          <p className="filename-line">
            <FileVideo2 size={15} />
            <span>{filename}</span>
          </p>
        </div>
        <Button variant="outline" asChild>
          <Link href="/analysis/new">
            <Plus size={17} />
            Новый анализ
          </Link>
        </Button>
      </div>
      {demo && (
        <div className="demo-banner">
          <Info size={16} />
          <p>Учебный пример. События и кадры подготовлены заранее.</p>
        </div>
      )}
      {!demo && job.warnings?.length ? (
        <div className="demo-banner" role="note">
          <Info size={16} />
          <div>
            <strong>Ограничения анализа</strong>
            {job.warnings.map((warning, i) => (
              <p key={i}>{warning}</p>
            ))}
          </div>
        </div>
      ) : null}
      <div className="results-grid">
        <div className="results-left">
          <VideoPlayer
            source={source}
            previewUrl={job.preview_url}
            events={events}
            selected={selected}
            onSelect={selectEvent}
            seek={seek}
            demo={!!preset}
            expectedFile={meta}
          />
          {selectedEvent ? (
            <EvidencePanel event={selectedEvent} demo={demo} onSeek={goTo} />
          ) : (
            <div className="panel no-evidence">
              <ImagePlaceholder />
              <div>
                <h3>Кадры событий отсутствуют</h3>
                <p>Система не отметила фрагментов для дополнительной проверки.</p>
              </div>
            </div>
          )}
        </div>
        <aside className="analysis-panel panel">
          <div className="analysis-panel-title">
            <ScanLine size={20} />
            <h2>Анализ событий</h2>
          </div>
          <div className={cn("verdict-block", !violation && "verdict-clear")}>
            {violation ? <TriangleAlert size={25} /> : <CircleCheck size={27} />}
            <h3>{violation ? "Требует внимания" : "Признаки не обнаружены"}</h3>
            <p>
              {violation
                ? "Проверьте отмеченные фрагменты."
                : "Модель не сработала на этой записи. Это не подтверждает проверку всех правил кассовой операции."}
            </p>
            <span>
              {violation ? `${events.length} события для проверки` : "0 отмеченных событий"}
            </span>
          </div>
          {violation ? (
            <div className="event-list">
              <div className="event-list-heading">
                <h3>События</h3>
              </div>
              {events.map((event, index) => (
                <button
                  className={cn("event-item", selected === index && "selected")}
                  key={index}
                  onClick={() => selectEvent(index)}
                  aria-pressed={selected === index}
                >
                  <span className="event-item-top">
                    <span className="event-index">
                      Событие {String(index + 1).padStart(2, "0")}
                    </span>
                    <span className="event-time">
                      <Clock3 size={12} />
                      {timecode(event.timecode_sec)}
                    </span>
                  </span>
                  <strong>{event.type}</strong>
                  {event.classification_status === "unclassified" ? <span>Тип нарушения не установлен</span> : null}
                  <span>{event.certainty === "suspected" ? "Подозрение — требует проверки" : event.certainty === "insufficient_data" ? "Недостаточно данных" : event.certainty === "confirmed" ? "Подтверждение указано сервером" : "Достоверность не указана"}</span>
                </button>
              ))}
            </div>
          ) : null}
          <div className="human-review-note">
            <ShieldCheck size={18} />
            <p>Итоговое решение принимает специалист.</p>
          </div>
        </aside>
      </div>
    </div>
  );
}
function ImagePlaceholder() {
  return <FileVideo2 size={26} />;
}
