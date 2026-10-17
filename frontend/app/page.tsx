import Link from "next/link";
import { ArrowRight, CirclePlay } from "lucide-react";
import { DemoList } from "@/components/dashboard/demo-list";
import { LastAnalysis } from "@/components/dashboard/last-analysis";
import { UploadForm } from "@/components/upload/upload-form";
export default function Dashboard() {
  return (
    <div className="page dashboard-page">
      <div className="page-heading workspace-heading">
        <div>
          <h1>Проверка видео</h1>
          <p>Загрузите запись кассовой операции.</p>
        </div>
        <Link className="text-link example-link" href="/analysis/demo-violation">
          <CirclePlay size={18} />
          Посмотреть пример
        </Link>
      </div>
      <UploadForm embedded />
      <LastAnalysis compact />
      <section className="demo-section" aria-labelledby="demo-title">
        <div className="section-heading">
          <h2 id="demo-title">Примеры проверки</h2>
          <Link href="/analysis" className="text-link">
            Все записи
            <ArrowRight size={16} />
          </Link>
        </div>
        <p className="section-description">Учебные данные, не результаты модели.</p>
        <DemoList compact />
      </section>
    </div>
  );
}
