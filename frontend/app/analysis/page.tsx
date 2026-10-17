import Link from "next/link";
import { Plus } from "lucide-react";
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
          <h2>Примеры проверки</h2>
          <span className="badge badge-neutral">Демо</span>
        </div>
        <p className="section-description">Учебные данные, не результаты модели.</p>
        <DemoList />
      </section>
    </div>
  );
}
