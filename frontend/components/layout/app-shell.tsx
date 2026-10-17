"use client";
import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  ArrowUpRight,
  ChevronRight,
  CircleHelp,
  Clapperboard,
  LayoutDashboard,
  Menu,
  Plus,
  ScanLine,
  ShieldCheck,
  X,
} from "lucide-react";
import { USE_MOCK_API } from "@/services/api";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);
  const sidebar = useRef<HTMLElement>(null);
  const menu = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    if (!open) return;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    sidebar.current?.querySelector<HTMLElement>("a")?.focus();
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setOpen(false);
        menu.current?.focus();
      }
      if (event.key !== "Tab") return;
      const items = Array.from(
        sidebar.current?.querySelectorAll<HTMLElement>("a[href], button:not([disabled])") ?? [],
      );
      const first = items[0],
        last = items[items.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last?.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first?.focus();
      }
    };
    document.addEventListener("keydown", onKey);
    return () => {
      document.body.style.overflow = previousOverflow;
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);
  const isResult = pathname.startsWith("/analysis/") && pathname !== "/analysis/new";
  const links = [
    { href: "/", label: "Обзор", icon: LayoutDashboard, active: pathname === "/" },
    {
      href: "/analysis/new",
      label: "Новый анализ",
      icon: Plus,
      active: pathname === "/analysis/new",
    },
    {
      href: "/analysis",
      label: "Результаты анализа",
      icon: Clapperboard,
      active: pathname === "/analysis" || isResult,
    },
  ];
  return (
    <div className="app-shell">
      <a href="#main-content" className="skip-link">
        Перейти к содержимому
      </a>
      {open && (
        <button
          className="sidebar-scrim"
          aria-label="Закрыть навигацию"
          onClick={() => setOpen(false)}
        />
      )}
      <aside
        ref={sidebar}
        className={cn("sidebar", open && "is-open")}
        role={open ? "dialog" : undefined}
        aria-modal={open || undefined}
        aria-label="Основная навигация"
      >
        <Link
          href="/"
          className="brand"
          onClick={() => setOpen(false)}
          aria-label="Forte Vision — главная"
        >
          <span className="brand-mark">
            <ScanLine size={26} strokeWidth={2.3} />
          </span>
          <span>
            forte<span className="brand-light">vision</span>
            <small>ВИДЕОКОНТРОЛЬ ОПЕРАЦИЙ</small>
          </span>
        </Link>
        <button className="mobile-close" onClick={() => setOpen(false)} aria-label="Закрыть меню">
          <X />
        </button>
        <div className="workspace">
          <span className="workspace-monogram">F</span>
          <div>
            ForteBank<small>Рабочее пространство</small>
          </div>
          <ShieldCheck size={16} />
        </div>
        <div className="nav-section-label">Рабочая область</div>
        <nav>
          {links.map(({ icon: Icon, ...item }) => (
            <Link
              key={item.href}
              href={item.href}
              className={cn("nav-link", item.active && "active")}
              aria-current={item.active ? "page" : undefined}
              onClick={() => setOpen(false)}
            >
              <Icon size={19} />
              <span>{item.label}</span>
              {item.active && <span className="nav-dot" />}
            </Link>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div className="local-note">
            <ShieldCheck size={22} />
            <strong>Данные под контролем</strong>
            <p>
              {USE_MOCK_API
                ? "В деморежиме видео остаётся в вашем браузере."
                : "Видео передаётся только в настроенный сервис анализа."}
            </p>
            <span>
              {USE_MOCK_API ? "Локальный деморежим" : "Защищённый серверный API"}
              <ArrowUpRight size={14} />
            </span>
          </div>
          <Dialog>
            <DialogTrigger asChild>
              <button className="help-button">
                <CircleHelp size={18} />
                Как это работает
                <ArrowUpRight size={15} />
              </button>
            </DialogTrigger>
            <DialogContent>
              <DialogTitle className="dialog-title">От видео к проверке события</DialogTitle>
              <DialogDescription className="muted">
                Forte Vision помогает специалисту находить фрагменты, требующие внимания.
              </DialogDescription>
              <ol className="help-steps">
                <li>
                  <strong>Загрузите запись</strong>
                  <p>Выберите один файл .mov. В деморежиме он не отправляется на сервер.</p>
                </li>
                <li>
                  <strong>Дождитесь результата</strong>
                  <p>
                    Интерфейс запрашивает статус задачи. Процент не отображается, если сервер его не
                    сообщает.
                  </p>
                </li>
                <li>
                  <strong>Проверьте контекст</strong>
                  <p>
                    Выберите событие, изучите видео и кадры. Окончательное решение принимает
                    специалист.
                  </p>
                </li>
              </ol>
              <div className="note">
                Демособытия и схематичные кадры созданы для проверки интерфейса. Они не являются
                выводами AI.
              </div>
            </DialogContent>
          </Dialog>
          <div className="sidebar-footer">
            <span className="project-avatar">FV</span>
            <div>
              Datathon 2026<small>Прототип команды</small>
            </div>
            <span className="version">v0.1</span>
          </div>
        </div>
      </aside>
      <div className="app-main" inert={open}>
        <header className="topbar">
          <div className="breadcrumbs">
            <Button
              ref={menu}
              variant="ghost"
              size="icon"
              className="menu-toggle"
              aria-label="Открыть меню"
              aria-expanded={open}
              onClick={() => setOpen(!open)}
            >
              <Menu size={21} />
            </Button>
            <span className="breadcrumb-root">Рабочая область</span>
            <ChevronRight size={14} className="breadcrumb-root" />
            <span>
              {isResult ? "Проверка записи" : (links.find((l) => l.active)?.label ?? "Страница")}
            </span>
          </div>
          <div className="topbar-right">
            <span className={cn("mode-badge", !USE_MOCK_API && "live-mode")}>
              <span className="status-dot" />
              {USE_MOCK_API ? "Деморежим" : "Режим API"}
            </span>
            <span className="header-divider" />
            <span className="reviewer-avatar" title="Рабочее пространство Forte Vision">
              FV
            </span>
          </div>
        </header>
        <main id="main-content" tabIndex={-1}>
          {children}
        </main>
        <footer className="page-footer">
          <span>
            Forte Vision <span className="footer-dot">·</span> Datathon 2026
          </span>
          <span>
            <ShieldCheck size={13} /> Решение остаётся за специалистом
          </span>
        </footer>
      </div>
    </div>
  );
}
