import Link from "next/link";
import { ArrowRight, ArrowUpRight, CirclePlay, FileVideo2, Plus } from "lucide-react";
import { DemoList } from "@/components/dashboard/demo-list";
import { LastAnalysis } from "@/components/dashboard/last-analysis";
import { USE_MOCK_API } from "@/services/api";
export default function Dashboard() {
  return (
    <div className="page dashboard-page">
      <section className="overview-intro">
        <div className="intro-copy">
          <h1>
            Проверка кассовых
            <br />
            <span>операций.</span>
          </h1>
          <p>
            Видеозапись, события и доказательства.
            <br />
            Всё, что нужно для внимательной проверки.
          </p>
          <Link className="text-link intro-demo-link" href="/analysis/demo-violation">
            <CirclePlay size={18} />
            Посмотреть пример
            <ArrowUpRight size={15} />
          </Link>
        </div>
        <Link className="quick-upload" href="/analysis/new" aria-label="Новый анализ">
          <div className="upload-file-outline">
            <FileVideo2 size={32} strokeWidth={1.2} />
            <span>.mov</span>
          </div>
          <div>
            <h2>Начните с видеозаписи</h2>
            <p>Одна касса. Одна операция.</p>
          </div>
          <span className="btn btn-primary">
            <Plus size={17} />
            Новый анализ
          </span>
          <small>MOV · до 500 МБ</small>
        </Link>
      </section>
      <div className="process-line" aria-label="Этапы проверки">
        <span>
          <span className="process-number">1</span>Добавьте запись
        </span>
        <ArrowRight size={14} />
        <span>
          <span className="process-number">2</span>Дождитесь анализа
        </span>
        <ArrowRight size={14} />
        <span>
          <span className="process-number">3</span>Проверьте события
        </span>
      </div>
      <section className="recent-section">
        <div className="section-heading">
          <h2>Ваше рабочее пространство</h2>
        </div>
        <LastAnalysis compact />
      </section>
      <section className="demo-section" aria-labelledby="demo-title">
        <div className="section-heading">
          <div className="inline-heading">
            <h2 id="demo-title">Учебные записи</h2>
            <span className="count-label">03</span>
          </div>
          <Link href="/analysis" className="text-link">
            Все записи
            <ArrowUpRight size={15} />
          </Link>
        </div>
        <p className="section-description">
          Готовые примеры для знакомства с проверкой. Данные демонстрационные.
        </p>
        <DemoList compact />
      </section>
      <div className="workspace-footnote">
        <span className="footnote-marker" />
        <p>
          {USE_MOCK_API
            ? "Сейчас вы в деморежиме. Загруженные видео остаются в браузере, результаты заданы заранее."
            : "Видео передаётся в настроенный сервис. Результат анализа требует проверки специалистом."}
        </p>
      </div>
    </div>
  );
}
