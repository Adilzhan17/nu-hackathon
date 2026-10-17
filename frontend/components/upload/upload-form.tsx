"use client";
import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import {
  ArrowLeft,
  ArrowRight,
  Check,
  FileVideo2,
  Info,
  LoaderCircle,
  LockKeyhole,
  ShieldCheck,
  Trash2,
  UploadCloud,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { analyzeVideo, USE_MOCK_API } from "@/services/api";
import { rememberUpload } from "@/lib/session";
import { cn, fileSize, validateVideo } from "@/lib/utils";
import type { MockScenario } from "@/types/analysis";
export function UploadForm() {
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
    try {
      const { job_id } = await analyzeVideo(file, scenario, controller.current.signal);
      if (controller.current.signal.aborted) return;
      rememberUpload(job_id, file, {
        name: file.name,
        size: file.size,
        createdAt: Date.now(),
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
    <div className="page upload-page">
      <Link className="back-link" href="/">
        <ArrowLeft size={16} />К обзору
      </Link>
      <div className="page-heading">
        <div>
          <h1>Новый анализ</h1>
          <p>Загрузите запись кассовой операции для проверки.</p>
        </div>
        <span className="subtle-tag">
          <LockKeyhole size={14} />
          {USE_MOCK_API ? "Без отправки на сервер" : "Ваш сервис обработки"}
        </span>
      </div>
      <div className="upload-layout">
        <form className="panel upload-panel" onSubmit={submit}>
          <div className="section-heading">
            <h2>Видеозапись</h2>
            <span>1 файл</span>
          </div>
          <p className="section-description">Выберите исходную запись с камеры наблюдения.</p>
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
                <p>или выберите файл на компьютере</p>
                <Button
                  variant="outline"
                  type="button"
                  onClick={() => input.current?.click()}
                  disabled={!ready}
                >
                  Выбрать файл
                  <ArrowUpIcon />
                </Button>
                <small>
                  .MOV <span>·</span> до 500 МБ
                </small>
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
          {USE_MOCK_API && (
            <fieldset className="scenario-fieldset" disabled={!ready || uploading}>
              <legend>
                Сценарий демонстрации <span className="badge badge-neutral">Демо</span>
              </legend>
              <p>Заранее заданный результат, не вывод AI по вашему видео.</p>
              <div className="scenario-options">
                {(
                  [
                    {
                      value: "violation",
                      label: "Есть события",
                      detail: "Два фрагмента для проверки",
                    },
                    {
                      value: "no_violation",
                      label: "Нет признаков",
                      detail: "Отрицательный результат",
                    },
                    { value: "failed", label: "Ошибка", detail: "Сбой обработки" },
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
                    <span>
                      <strong>{item.label}</strong>
                      <small>{item.detail}</small>
                    </span>
                  </label>
                ))}
              </div>
            </fieldset>
          )}
          <div className="upload-submit">
            <span>
              <ShieldCheck size={16} />
              {USE_MOCK_API ? "Видео останется на устройстве" : "Отправка в ваш сервис"}
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
          <div role="status" className="sr-only">
            {uploading ? "Видеозапись загружается. Дождитесь завершения." : ""}
          </div>
        </form>
        <aside className="upload-guide">
          <h2>Перед загрузкой</h2>
          <p>Выберите полную запись операции, чтобы сохранить последовательность действий.</p>
          <ul>
            <li>
              <Check size={16} />
              <span>Одна камера и одна касса в кадре</span>
            </li>
            <li>
              <Check size={16} />
              <span>Сохранена последовательность операции</span>
            </li>
            <li>
              <Check size={16} />
              <span>Действия и предметы различимы</span>
            </li>
          </ul>
          <div className="guide-divider" />
          <h3>Что будет после загрузки?</h3>
          <p>Вы увидите статус обработки, затем результат с таймкодами и кадрами для проверки.</p>
          <div className="note">
            <Info size={18} />
            <p>
              Некоторые кодеки MOV не поддерживаются браузером. Для таких записей понадобится
              совместимая preview-версия от бэкенда.
            </p>
          </div>
        </aside>
      </div>
    </div>
  );
}
function ArrowUpIcon() {
  return <UploadCloud size={16} />;
}
