"use client";
import { useEffect, useRef, useState } from "react";
import { FileVideo2, Maximize, Pause, Play, RotateCcw, VolumeX } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn, timecode, validateVideo } from "@/lib/utils";
import type { Violation } from "@/types/analysis";
export function VideoPlayer({
  source,
  previewUrl,
  events,
  seek,
  selected,
  onSelect,
  demo,
  expectedFile,
}: {
  source?: string;
  previewUrl?: string;
  events: Violation[];
  seek?: { seconds: number; nonce: number };
  selected: number;
  onSelect: (index: number) => void;
  demo: boolean;
  expectedFile?: { name: string; size: number };
}) {
  const video = useRef<HTMLVideoElement>(null);
  const container = useRef<HTMLDivElement>(null);
  const replacement = useRef<string | undefined>(undefined);
  const [localSource, setLocalSource] = useState<string>();
  const [usePreview, setUsePreview] = useState(false);
  const [failed, setFailed] = useState(false);
  const [message, setMessage] = useState("");
  const [playing, setPlaying] = useState(false);
  const [duration, setDuration] = useState(0);
  const [current, setCurrent] = useState(0);
  const src = usePreview ? previewUrl : (localSource ?? source ?? previewUrl);
  const pendingSeek = useRef<number | null>(null);
  const seekSeconds = seek?.seconds;
  const seekNonce = seek?.nonce;
  useEffect(() => {
    if (seekSeconds === undefined) return;
    if (video.current && video.current.readyState >= 1)
      video.current.currentTime = Math.min(seekSeconds, video.current.duration || seekSeconds);
    else pendingSeek.current = seekSeconds;
  }, [seekSeconds, seekNonce]);
  useEffect(
    () => () => {
      if (replacement.current) URL.revokeObjectURL(replacement.current);
    },
    [],
  );
  function handleError() {
    if (previewUrl && src !== previewUrl) {
      setUsePreview(true);
      return;
    }
    setFailed(true);
    setPlaying(false);
  }
  async function toggle() {
    if (!video.current || !src || failed) return;
    if (video.current.paused) {
      try {
        await video.current.play();
      } catch {
        setMessage(
          "Не удалось начать воспроизведение. Попробуйте снова или выберите совместимую запись.",
        );
      }
    } else video.current.pause();
  }
  function selectReplacement(file?: File) {
    if (!file) return;
    const issue = validateVideo(file);
    if (issue) {
      setMessage(issue);
      return;
    }
    if (expectedFile && (file.name !== expectedFile.name || file.size !== expectedFile.size)) {
      setMessage(
        "Выберите исходный файл с тем же именем и размером, чтобы таймкоды соответствовали результату.",
      );
      return;
    }
    if (replacement.current) URL.revokeObjectURL(replacement.current);
    replacement.current = URL.createObjectURL(file);
    setLocalSource(replacement.current);
    setUsePreview(false);
    setFailed(false);
    setMessage("");
  }
  return (
    <div className="video-card panel" ref={container}>
      <div className="video-topline">
        <span>
          <FileVideo2 size={16} />
          Видеозапись
        </span>
        <span>
          {demo ? "Учебная схема" : "Исходная запись"}
          <VolumeX size={14} />
        </span>
      </div>
      <div className="video-stage">
        {src && !failed ? (
          <video
            key={src}
            ref={video}
            src={src}
            muted
            playsInline
            preload="metadata"
            onError={handleError}
            onLoadedMetadata={() => {
              const node = video.current!;
              setDuration(Number.isFinite(node.duration) ? node.duration : 0);
              if (pendingSeek.current !== null) {
                node.currentTime = Math.min(pendingSeek.current, node.duration);
                pendingSeek.current = null;
              }
            }}
            onTimeUpdate={() => setCurrent(video.current?.currentTime ?? 0)}
            onPlay={() => setPlaying(true)}
            onPause={() => setPlaying(false)}
            onEnded={() => setPlaying(false)}
            aria-label="Запись кассовой операции, без звука"
          />
        ) : (
          <div className="video-unavailable">
            <FileVideo2 size={34} />
            <h3>{failed ? "Видео не воспроизводится" : "Исходная запись не открыта"}</h3>
            <p>
              {failed
                ? "Браузер не поддерживает кодек или файл недоступен. Бэкенд может вернуть совместимую MP4 preview-версию."
                : "После перезагрузки браузер теряет доступ к локальному файлу. Выберите ту же запись — повторный анализ не нужен."}
            </p>
            <label className="btn btn-outline">
              Выбрать исходный .mov
              <input
                className="sr-only"
                type="file"
                accept=".mov"
                aria-label="Повторно выбрать исходное видео"
                onChange={(e) => selectReplacement(e.target.files?.[0])}
              />
            </label>
          </div>
        )}
        {src && !failed && !playing && (
          <button className="video-play-overlay" onClick={toggle} aria-label="Воспроизвести видео">
            <Play size={26} fill="currentColor" />
          </button>
        )}
        {src && !failed && (
          <span className="video-source-label">
            {demo ? "ДЕМО · СХЕМА КАССОВОЙ ЗОНЫ" : "ЛОКАЛЬНАЯ ВИДЕОЗАПИСЬ"}
          </span>
        )}
      </div>
      <div className="video-controls">
        <Button
          variant="ghost"
          size="icon"
          aria-label={playing ? "Пауза" : "Воспроизвести"}
          onClick={toggle}
          disabled={!src || failed}
        >
          {playing ? <Pause size={19} /> : <Play size={19} />}
        </Button>
        <Button
          variant="ghost"
          size="icon"
          aria-label="Назад на 10 секунд"
          onClick={() => {
            if (video.current)
              video.current.currentTime = Math.max(0, video.current.currentTime - 10);
          }}
          disabled={!src || failed}
        >
          <RotateCcw size={17} />
        </Button>
        <span className="player-time">
          {timecode(current)} <span>/ {timecode(duration)}</span>
        </span>
        <span className="controls-spacer" />
        <span className="muted-audio">
          <VolumeX size={15} />
          Без звука
        </span>
        <Button
          variant="ghost"
          size="icon"
          aria-label="Полноэкранный режим"
          disabled={!src || failed}
          onClick={() => {
            if (document.fullscreenElement) void document.exitFullscreen();
            else
              void container.current
                ?.requestFullscreen()
                .catch(() => setMessage("Полноэкранный режим недоступен в этом браузере."));
          }}
        >
          <Maximize size={17} />
        </Button>
      </div>
      <div className="player-timeline">
        <input
          aria-label="Позиция воспроизведения в секундах"
          type="range"
          min={0}
          max={duration || 90}
          step={0.1}
          value={current}
          disabled={!src || failed || !duration}
          onChange={(e) => {
            const seconds = Number(e.target.value);
            setCurrent(seconds);
            if (video.current) video.current.currentTime = seconds;
          }}
        />
        <div className="event-markers">
          {events.map((event, index) => (
            <button
              key={index}
              className={cn("event-marker", selected === index && "selected")}
              style={{
                left: `${Math.max(2, Math.min(98, (event.timecode_sec / (duration || Math.max(90, ...events.map((e) => e.timecode_sec)))) * 100))}%`,
              }}
              aria-label={`Событие ${index + 1}, ${timecode(event.timecode_sec)}`}
              aria-pressed={selected === index}
              onClick={() => onSelect(index)}
            />
          ))}
        </div>
        <div className="timeline-labels">
          <span>00:00</span>
          <span>{timecode((duration || 90) / 3)}</span>
          <span>{timecode(((duration || 90) * 2) / 3)}</span>
          <span>{timecode(duration || 90)}</span>
        </div>
      </div>
      <div className="player-legend">
        <span>
          <i />
          Отмеченные события
        </span>
        <span>Нажмите на метку, чтобы перейти к фрагменту</span>
      </div>
      {message && (
        <p className="player-message" role="alert">
          {message}
        </p>
      )}
    </div>
  );
}
