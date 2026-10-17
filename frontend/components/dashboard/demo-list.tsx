"use client";
import { useState } from "react";
import Link from "next/link";
import { ArrowUpRight, CircleAlert, CircleCheck, Search, TriangleAlert, X } from "lucide-react";
import { DEMO_JOBS } from "@/services/mock-api";
import { cn, timecode } from "@/lib/utils";
type Filter = "all" | "violation" | "no_violation" | "failed";
export function DemoList({ compact = false }: { compact?: boolean }) {
  const [filter, setFilter] = useState<Filter>("all");
  const [query, setQuery] = useState("");
  const records = DEMO_JOBS.filter(
    (job) =>
      (filter === "all" || job.scenario === filter) &&
      `${job.name} ${job.label}`
        .toLocaleLowerCase("ru")
        .includes(query.toLocaleLowerCase("ru").trim()),
  );
  return (
    <div className={cn("record-library", compact && "compact-library")}>
      {!compact && (
        <div className="library-toolbar">
          <div className="record-filters" role="group" aria-label="Фильтр результатов">
            {(
              [
                { value: "all", label: "Все записи" },
                { value: "violation", label: "С событиями" },
                { value: "no_violation", label: "Без событий" },
                { value: "failed", label: "С ошибкой" },
              ] as const
            ).map((item) => (
              <button
                key={item.value}
                className={cn(filter === item.value && "active")}
                aria-pressed={filter === item.value}
                onClick={() => setFilter(item.value)}
              >
                {item.label}
                <span>
                  {item.value === "all"
                    ? DEMO_JOBS.length
                    : DEMO_JOBS.filter((job) => job.scenario === item.value).length}
                </span>
              </button>
            ))}
          </div>
          <label className="record-search">
            <Search size={16} />
            <input
              type="search"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Найти запись…"
              aria-label="Найти запись"
            />
            {query && (
              <button type="button" onClick={() => setQuery("")} aria-label="Очистить поиск">
                <X size={14} />
              </button>
            )}
          </label>
        </div>
      )}
      <div className="demo-table">
        <div className="demo-table-head">
          <span>Название записи</span>
          <span>Длительность</span>
          <span>Результат проверки</span>
          <span />
        </div>
        {records.map((job) => {
          const Icon =
            job.scenario === "violation"
              ? TriangleAlert
              : job.scenario === "no_violation"
                ? CircleCheck
                : CircleAlert;
          return (
            <Link className="demo-row" href={`/analysis/${job.id}`} key={job.id}>
              <div className="file-cell">
                <div className={cn("record-preview", `preview-${job.scenario}`)}>
                  <img src="/demo/scene.svg" alt="" />
                  <span>MOV</span>
                </div>
                <div>
                  <strong>{job.name}</strong>
                </div>
              </div>
              <span className="duration-cell">{timecode(job.duration)}</span>
              <div className="result-cell">
                <span
                  className={cn(
                    "badge",
                    job.scenario === "violation"
                      ? "badge-warning"
                      : job.scenario === "no_violation"
                        ? "badge-success"
                        : "badge-error",
                  )}
                >
                  <Icon size={13} />
                  {job.label}
                </span>
              </div>
              <ArrowUpRight size={18} className="row-arrow" />
            </Link>
          );
        })}
        {records.length === 0 && (
          <div className="search-empty" role="status">
            <Search size={24} />
            <h3>Запись не найдена</h3>
            <p>Измените запрос или выберите другой результат.</p>
            <button
              className="text-link"
              onClick={() => {
                setQuery("");
                setFilter("all");
              }}
            >
              Показать все записи
              <ArrowUpRight size={14} />
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
