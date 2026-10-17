import Link from "next/link";
import { Plus, Info } from "lucide-react";
import { DemoList } from "@/components/dashboard/demo-list";
import { Button } from "@/components/ui/button";
import { LastAnalysis } from "@/components/dashboard/last-analysis";
export const metadata = { title: "Результаты анализа" };
export default function AnalysisPage() {
  return (
    <div className="page">
      <div className="page-heading">
        <div>
          <h1>Результаты анализа</h1>
          <p>Откройте последнюю задачу или изучите учебный пример.</p>
        </div>
        <Button asChild>
          <Link href="/analysis/new">
            <Plus size={18} />
            Новый анализ
          </Link>
        </Button>
      </div>
      <LastAnalysis />
      <section className="examples-panel">
        <div className="section-heading">
          <h2>Учебные записи</h2>
          <span className="badge badge-neutral">Демо</span>
        </div>
        <DemoList />
      </section>
      <div className="note">
        <Info size={18} />
        <p>
          Примеры содержат заранее подготовленные события и схематичное видео. Они показывают
          возможности интерфейса и не являются результатом работы модели.
        </p>
      </div>
    </div>
  );
}
