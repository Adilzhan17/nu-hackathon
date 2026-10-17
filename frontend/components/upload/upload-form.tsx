"use client";
import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import {
  ArrowRight,
  Check,
  FileVideo2,
  Info,
  LoaderCircle,
  Trash2,
  UploadCloud,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { analyzeVideo, USE_MOCK_API } from "@/services/api";
import { rememberUpload } from "@/lib/session";
import { cn, fileSize, validateVideo } from "@/lib/utils";
import type { MockScenario } from "@/types/analysis";
export function UploadForm({ embedded = false }: { embedded?: boolean }) {
  const router = useRouter();
  const input = useRef<HTMLInputElement>(null);
  const controller = useRef<AbortController | null>(null);
  const [ready, setReady] = useState(false);
  const [file, setFile] = useState<File>();
  const [error, setError] = useState("");
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [scenario, setScenario] = useState<MockScenario>("violation");
  useEffect(() => {
    setReady(true);
    return () => controller.current?.abort();
  }, []);
  function selectFile(files: FileList | null) {
    setDragging(false);
    if (!files?.length) return;
    if (files.length > 1) {
      setError("За один раз можно проанализировать одну запись. Выберите один .mov файл.");
      return;
    }
    const selected = files[0];
    const issue = validateVideo(selected);
    if (issue) {
      setError(issue);
      return;
    }
    setFile(selected);
    setError("");
  }
  async function submit(event: React.FormEvent) {
    event.preventDefault();
    if (!file) {
      setError("Сначала выберите видеофайл в формате .mov.");
      input.current?.focus();
      return;
    }
    if (uploading) return;
    setUploading(true);
    setError("");
    controller.current = new AbortController();
    const createdAt = Date.now();
    try {
      const { job_id } = await analyzeVideo(file, scenario, controller.current.signal);
      if (controller.current.signal.aborted) return;
      rememberUpload(job_id, file, {
        name: file.name,
        size: file.size,
        createdAt,
        mock: USE_MOCK_API,
        scenario,
      });
      router.push(`/analysis/${encodeURIComponent(job_id)}`);
    } catch (e) {
      if (!controller.current?.signal.aborted) {
        setError(e instanceof Error ? e.message : "Не удалось загрузить видео.");
        setUploading(false);
      }
    }
  }
  return (
    <div className={embedded ? "upload-workspace" : "page upload-page"}>
      {!embedded && (
        <div className="page-heading">
          <div>
            <h1>Новый анализ</h1>
            <p>Загрузите запись кассовой операции.</p>
          </div>
        </div>
      )}
      <form className="upload-panel" onSubmit={submit}>
        <div
          className={cn("dropzone", dragging && "dragging", file && "has-file")}
          onDragOver={(e) => {
            e.preventDefault();
            if (ready && !uploading) setDragging(true);
          }}
          onDragLeave={(e) => {
            if (!e.currentTarget.contains(e.relatedTarget as Node)) setDragging(false);
          }}
          onDrop={(e) => {
            e.preventDefault();
            if (ready && !uploading) selectFile(e.dataTransfer.files);
          }}
        >
          {file ? (
            <>
              <span className="upload-symbol selected">
                <FileVideo2 size={32} />
              </span>
              <strong className="selected-filename">{file.name}</strong>
              <p>
                {fileSize(file.size)} <span>·</span> MOV
              </p>
              <span className="valid-file">
                <Check size={14} />
                Файл готов к загрузке
              </span>
              <div className="selected-actions">
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={() => input.current?.click()}
                  disabled={!ready || uploading}
                >
                  Заменить файл
                </Button>
                <Button
                  type="button"
                  variant="ghost"
                  size="icon"
                  aria-label="Удалить выбранный файл"
                  disabled={!ready || uploading}
                  onClick={() => {
                    setFile(undefined);
                    setError("");
                    if (input.current) input.current.value = "";
                  }}
                >
                  <Trash2 size={17} />
                </Button>
              </div>
            </>
          ) : (
            <>
              <span className="upload-symbol">
                <UploadCloud size={33} strokeWidth={1.5} />
              </span>
              <h3>Перетащите видео сюда</h3>
              <p>MOV · до 64 МиБ</p>
              <Button type="button" onClick={() => input.current?.click()} disabled={!ready}>
                Выбрать файл
                <ArrowUpIcon />
              </Button>
            </>
          )}
          <input
            ref={input}
            className="sr-only"
            type="file"
            accept=".mov,video/quicktime"
            aria-label="Выбрать видео .mov"
            aria-describedby={error ? "upload-error" : undefined}
            disabled={!ready || uploading}
            onChange={(e) => {
              selectFile(e.target.files);
              e.target.value = "";
            }}
          />
        </div>
        {error && (
          <div id="upload-error" className="error-message" role="alert">
            <Info size={18} />
            <span>{error}</span>
          </div>
        )}
        {file && (
          <div className="upload-submit">
            <span>
              {USE_MOCK_API
                ? "Демо · видео остаётся в браузере"
                : "Видео будет отправлено на анализ"}
            </span>
            <Button type="submit" disabled={!ready || uploading}>
              {uploading ? (
                <>
                  <LoaderCircle size={18} className="spin" />
                  {USE_MOCK_API ? "Подготовка…" : "Загрузка…"}
                </>
              ) : (
                <>
                  Начать анализ
                  <ArrowRight size={17} />
                </>
              )}
            </Button>
          </div>
        )}
        {USE_MOCK_API && (
          <div className="demo-upload-note">Деморежим: результат задан заранее.</div>
        )}
        <div className="upload-options">
          <details className="disclosure">
            <summary>Требования к записи</summary>
            <div className="disclosure-body">
              <p>Одна камера и одна касса. Полная операция, различимые действия и предметы.</p>
              <p>
                MOV — контейнер: воспроизведение зависит от кодека. Если видео не открывается, нужна
                совместимая версия от сервиса анализа.
              </p>
            </div>
          </details>
          {USE_MOCK_API && (
            <details className="disclosure demo-settings">
              <summary>Настроить демосценарий</summary>
              <fieldset className="scenario-fieldset" disabled={!ready || uploading}>
                <legend className="sr-only">Сценарий демонстрации</legend>
                <div className="scenario-options">
                  {(
                    [
                      { value: "violation", label: "Есть события" },
                      { value: "no_violation", label: "Нет признаков" },
                      { value: "failed", label: "Ошибка" },
                    ] as const
                  ).map((item) => (
                    <label
                      className={cn("scenario-option", scenario === item.value && "selected")}
                      key={item.value}
                    >
                      <input
                        type="radio"
                        name="scenario"
                        value={item.value}
                        checked={scenario === item.value}
                        onChange={() => setScenario(item.value)}
                      />
                      <span>{item.label}</span>
                    </label>
                  ))}
                </div>
              </fieldset>
            </details>
          )}
        </div>
        <div role="status" className="sr-only">
          {uploading ? "Видеозапись загружается. Дождитесь завершения." : ""}
        </div>
      </form>
    </div>
  );
}
function ArrowUpIcon() {
  return <UploadCloud size={16} />;
}
