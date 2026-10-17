"use client";
import { useEffect, useState } from "react";
import Link from "next/link";
import { ArrowRight, FileVideo2 } from "lucide-react";
import { getMetadata } from "@/lib/session";
export function LastAnalysis() {
  const [last, setLast] = useState<{ id: string; name: string }>();
  useEffect(() => {
    try {
      const id = sessionStorage.getItem("forte:last-job");
      const meta = id ? getMetadata(id) : undefined;
      if (id && meta) setLast({ id, name: meta.name });
    } catch {
      /* Session storage is optional. */
    }
  }, []);
  return last ? (
    <Link className="last-analysis panel" href={`/analysis/${encodeURIComponent(last.id)}`}>
      <FileVideo2 size={24} />
      <span>
        <small>Последняя задача в этой вкладке</small>
        <strong>{last.name}</strong>
      </span>
      <span>Открыть результат</span>
      <ArrowRight size={18} />
    </Link>
  ) : (
    <div className="empty-session">
      <FileVideo2 size={24} />
      <div>
        <strong>В этой вкладке пока нет ваших анализов</strong>
        <p>После загрузки здесь появится ссылка на последнюю задачу.</p>
      </div>
    </div>
  );
}
