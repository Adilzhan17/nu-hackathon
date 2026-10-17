"use client";
import { useEffect, useState } from "react";
import { getJob, ApiError } from "@/services/api";
import type { AnalysisJob } from "@/types/analysis";
import { getMetadata } from "@/lib/session";
export function useAnalysis(id: string) {
  const [job, setJob] = useState<AnalysisJob>();
  const [error, setError] = useState<string>();
  const [attempt, setAttempt] = useState(0);
  const [retrying, setRetrying] = useState(false);
  useEffect(() => {
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    let failures = 0;
    const started = getMetadata(id)?.createdAt ?? Date.now();
    setJob(undefined);
    setError(undefined);
    setRetrying(false);
    async function poll() {
      try {
        const next = await getJob(id, controller.signal);
        if (controller.signal.aborted) return;
        failures = 0;
        setRetrying(false);
        setJob(next);
        if (next.status !== "processing") return;
        if (Date.now() - started >= 300_000) {
          setError(
            "Анализ занимает больше пяти минут. Можно снова запросить результат — повторная загрузка не нужна.",
          );
          return;
        }
        timer = setTimeout(poll, 1800);
      } catch (e) {
        if (controller.signal.aborted) return;
        failures++;
        if (
          Date.now() - started < 300_000 &&
          failures < 3 &&
          !(e instanceof ApiError && e.status && e.status < 500 && e.status !== 429)
        ) {
          setRetrying(true);
          timer = setTimeout(poll, failures * 2000);
          return;
        }
        setRetrying(false);
        setError(e instanceof Error ? e.message : "Не удалось получить результат.");
      }
    }
    void poll();
    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, [id, attempt]);
  return { job, error, retrying, retry: () => setAttempt((a) => a + 1) };
}
