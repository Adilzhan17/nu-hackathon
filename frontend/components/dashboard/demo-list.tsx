import Link from "next/link";
import { ArrowUpRight, CircleAlert, CircleCheck, FileVideo2, TriangleAlert } from "lucide-react";
import { DEMO_JOBS } from "@/services/mock-api";
import { cn, timecode } from "@/lib/utils";
export function DemoList({ compact = false }: { compact?: boolean }) {
  return (
    <div className={cn("demo-table", compact && "compact-table")}>
      <div className="demo-table-head">
        <span>Видеозапись</span>
        <span>Длительность</span>
        <span>Результат</span>
        <span />
      </div>
      {DEMO_JOBS.map((job) => {
        const Icon =
          job.scenario === "violation"
            ? TriangleAlert
            : job.scenario === "no_violation"
              ? CircleCheck
              : CircleAlert;
        return (
          <Link className="demo-row" href={`/analysis/${job.id}`} key={job.id}>
            <div className="file-cell">
              <span className="file-icon">
                <FileVideo2 size={21} />
              </span>
              <div>
                <strong>{job.name}</strong>
                <small>
                  Учебный пример <span>·</span> .MOV
                </small>
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
              <small>{job.note}</small>
            </div>
            <ArrowUpRight size={18} className="row-arrow" />
          </Link>
        );
      })}
    </div>
  );
}
