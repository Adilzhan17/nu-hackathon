import Link from "next/link";
import {
  ArrowRight,
  ArrowUpRight,
  Check,
  CirclePlay,
  FileVideo2,
  Focus,
  LockKeyhole,
  Plus,
  ScanLine,
  ShieldCheck,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { DemoList } from "@/components/dashboard/demo-list";
import { USE_MOCK_API } from "@/services/api";
export default function Dashboard() {
  return (
    <div className="page dashboard-page">
      <div className="page-heading">
        <div>
          <h1>Обзор платформы</h1>
          <p>Видеоконтроль кассовых операций в одном рабочем пространстве.</p>
        </div>
        <span className="subtle-tag">ForteBank · Datathon 2026</span>
      </div>
      <section className="overview-hero">
        <div className="hero-copy">
          <h2>
            Внимание к каждой
            <br />
            <span>операции.</span>
          </h2>
          <p>
            Загрузите запись. Найдите события, требующие внимания. Проверьте детали — с опорой на
            видео.
          </p>
          <div className="hero-actions">
            <Button asChild>
              <Link href="/analysis/new">
                <Plus size={18} />
                Новый анализ
                <ArrowRight size={17} />
              </Link>
            </Button>
            <Link className="text-link" href="/analysis/demo-violation">
              <CirclePlay size={18} />
              Посмотреть пример
            </Link>
          </div>
          <div className="hero-assurance">
            <LockKeyhole size={14} />
            {USE_MOCK_API
              ? "Деморежим: видео остаётся в браузере"
              : "Видео обрабатывается вашим сервисом"}
          </div>
        </div>
        <div
          className="hero-visual"
          aria-label="Схема анализа: видеозапись, события, проверка специалистом"
        >
          <div className="visual-topline">
            <span>
              <span className="status-dot" />
              СЦЕНАРИЙ ПРОВЕРКИ
            </span>
            <ScanLine size={19} />
          </div>
          <div className="analysis-diagram">
            <div className="diagram-source">
              <FileVideo2 size={29} />
              <span>Запись камеры</span>
              <small>.MOV</small>
            </div>
            <div className="diagram-connector">
              <span />
              <ArrowRight size={15} />
            </div>
            <div className="diagram-focus">
              <Focus size={40} strokeWidth={1.4} />
              <span>Анализ событий</span>
              <div className="focus-corner tl" />
              <div className="focus-corner tr" />
              <div className="focus-corner bl" />
              <div className="focus-corner br" />
            </div>
          </div>
          <div className="visual-event">
            <span className="event-stem" />
            <span className="event-timestamp">00:24</span>
            <span>Событие для проверки</span>
            <ArrowUpRight size={16} />
          </div>
          <div className="visual-timeline">
            <div className="timeline-track">
              <span style={{ left: "27%" }} />
              <span style={{ left: "65%" }} />
            </div>
            <div>
              <span>00:00</span>
              <span>01:30</span>
            </div>
          </div>
          <span className="visual-caption">Иллюстрация · не результат анализа</span>
        </div>
      </section>
      <section className="workflow-section" aria-labelledby="workflow-title">
        <div className="section-heading">
          <h2 id="workflow-title">От записи к обоснованному решению</h2>
          <span>Три простых шага</span>
        </div>
        <div className="workflow">
          <div className="workflow-item">
            <span className="step-number">01</span>
            <div>
              <h3>Загрузите видео</h3>
              <p>
                Одна запись кассовой операции
                <br className="desktop-break" /> в формате .mov.
              </p>
            </div>
            <FileVideo2 size={22} />
          </div>
          <div className="workflow-item">
            <span className="step-number">02</span>
            <div>
              <h3>Дождитесь анализа</h3>
              <p>
                Система выделит события
                <br className="desktop-break" /> и соответствующие кадры.
              </p>
            </div>
            <ScanLine size={22} />
          </div>
          <div className="workflow-item">
            <span className="step-number">03</span>
            <div>
              <h3>Проверьте детали</h3>
              <p>
                Перейдите к таймкоду
                <br className="desktop-break" /> и оцените контекст.
              </p>
            </div>
            <ShieldCheck size={22} />
          </div>
        </div>
      </section>
      <section className="demo-section" aria-labelledby="demo-title">
        <div className="section-heading">
          <div className="inline-heading">
            <h2 id="demo-title">Познакомьтесь с результатами</h2>
            <span className="badge badge-neutral">Демо</span>
          </div>
          <Link href="/analysis" className="text-link">
            Все примеры
            <ArrowRight size={15} />
          </Link>
        </div>
        <p className="section-description">
          Три учебных сценария, чтобы изучить интерфейс до первой загрузки.
        </p>
        <DemoList compact />
      </section>
      <div className="privacy-strip">
        <div>
          <ShieldCheck size={22} />
          <span>
            <strong>Приватность заложена в процесс</strong>
            <small>Без распознавания лиц и анализа аудио. Без облачных AI-сервисов.</small>
          </span>
        </div>
        <span className="privacy-detail">
          <Check size={15} />
          Проверка человеком
        </span>
      </div>
    </div>
  );
}
