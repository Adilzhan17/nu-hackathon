"use client";
import { ArrowUpRight, BookOpen, Expand, ImageIcon, Info } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { timecode } from "@/lib/utils";
import type { Violation } from "@/types/analysis";
export function RegulatorySource({ event }: { event: Violation }) {
  const source = event.regulatory_source;
  return (
    <div className="regulatory-source">
      <BookOpen size={18} />
      <div>
        <h4>Нормативное основание</h4>
        {source ? (
          <>
            <p>
              {source.title}
              {source.section ? ` · ${source.section}` : ""}
            </p>
            {source.url && (
              <a className="text-link" href={source.url} target="_blank" rel="noopener noreferrer">
                Открыть источник
                <ArrowUpRight size={14} />
              </a>
            )}
          </>
        ) : (
          <p>Подтверждённый источник не указан.</p>
        )}
      </div>
    </div>
  );
}
export function EvidencePanel({
  event,
  demo,
  onSeek,
}: {
  event: Violation;
  demo: boolean;
  onSeek: (time: number) => void;
}) {
  return (
    <section className="panel evidence-panel">
      <div className="section-heading">
        <h2>
          <ImageIcon size={18} />
          Кадры события
        </h2>
        <span>
          {event.evidence_frames.length} {event.evidence_frames.length === 1 ? "кадр" : "кадра"}
        </span>
      </div>
      <div className="evidence-layout">
        <div className="evidence-frames">
          {event.evidence_frames.map((frame, i) => (
            <Dialog key={`${event.timecode_sec}-${i}`}>
              <DialogTrigger asChild>
                <button
                  className="evidence-thumbnail"
                  aria-label={`Увеличить кадр ${timecode(frame.timestamp_sec)}`}
                >
                  <img
                    src={`data:image/jpeg;base64,${frame.image_base64}`}
                    alt={`${demo ? "Демонстрационная схема" : "Кадр события"} на ${timecode(frame.timestamp_sec)}`}
                  />
                  <span className="frame-time">{timecode(frame.timestamp_sec)}</span>
                  <span className="frame-expand">
                    <Expand size={16} />
                  </span>
                </button>
              </DialogTrigger>
              <DialogContent className="evidence-dialog">
                <DialogTitle className="dialog-title">{event.type}</DialogTitle>
                <DialogDescription className="muted">
                  {timecode(frame.timestamp_sec)} ·{" "}
                  {demo
                    ? "Демонстрационная схема, не кадр загруженного видео"
                    : "Кадр, возвращённый сервисом анализа"}
                </DialogDescription>
                <img
                  className="evidence-full"
                  src={`data:image/jpeg;base64,${frame.image_base64}`}
                  alt={`Увеличенный кадр события ${timecode(frame.timestamp_sec)}`}
                />
                <p className="evidence-description">{event.description}</p>
                <RegulatorySource event={event} />
              </DialogContent>
            </Dialog>
          ))}
        </div>
        <div className="evidence-copy">
          <p>{event.description}</p>
          <button className="text-link" onClick={() => onSeek(event.timecode_sec)}>
            Перейти к {timecode(event.timecode_sec)}
            <ArrowUpRight size={14} />
          </button>
        </div>
      </div>
      {demo && (
        <p className="evidence-demo-note">
          <Info size={14} />
          Учебная схема. Не извлечена из загруженного видео.
        </p>
      )}
      <details className="disclosure evidence-source">
        <summary>Нормативное основание</summary>
        <RegulatorySource event={event} />
      </details>
    </section>
  );
}
