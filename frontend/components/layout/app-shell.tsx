"use client";
import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  ArrowUpRight,
  ChevronRight,
  CircleHelp,
  Clapperboard,
  Home,
  Menu,
  PanelLeftClose,
  PanelLeftOpen,
  Plus,
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

function HelpDialog() {
  return (
    <Dialog>
      <DialogTrigger asChild>
        <button className="help-button" title="Как это работает">
          <CircleHelp size={18} />
          <span>Помощь</span>
          <ArrowUpRight size={14} />
        </button>
      </DialogTrigger>
      <DialogContent>
        <DialogTitle className="dialog-title">Как проходит проверка</DialogTitle>
        <DialogDescription className="muted">
          От видеозаписи до события, которое можно проверить.
        </DialogDescription>
        <ol className="help-steps">
          <li>
            <strong>Добавьте запись</strong>
            <p>Один файл .mov до 500 МБ. В деморежиме он остаётся в браузере.</p>
          </li>
          <li>
            <strong>Дождитесь анализа</strong>
            <p>
              Статус обновляется автоматически. В деморежиме результат заранее задан выбранным
              сценарием.
            </p>
          </li>
          <li>
            <strong>Проверьте наблюдения</strong>
            <p>
              Выберите событие, перейдите к таймкоду и изучите кадры. Итоговое решение остаётся за
              вами.
            </p>
          </li>
        </ol>
        <div className="note">Учебные события и схемы не являются выводами AI по вашей записи.</div>
      </DialogContent>
    </Dialog>
  );
}

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);
  const [collapsed, setCollapsed] = useState(false);
  const sidebar = useRef<HTMLElement>(null);
  const menu = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    try {
      setCollapsed(localStorage.getItem("forte:sidebar-collapsed") === "true");
    } catch {
      /* Optional preference. */
    }
  }, []);
  useEffect(() => {
    if (!open) return;
    const overflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    sidebar.current?.querySelector<HTMLElement>("a")?.focus();
    const onKey = (event: KeyboardEvent) => {
      if (document.querySelector('[data-state="open"][role="dialog"]')) return;
      if (event.key === "Escape") setOpen(false);
      if (event.key !== "Tab") return;
      const items = Array.from(
        sidebar.current?.querySelectorAll<HTMLElement>("a[href], button:not([disabled])") ?? [],
      ).filter((el) => el.getClientRects().length && getComputedStyle(el).visibility !== "hidden");
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
    const onResize = () => {
      if (window.innerWidth > 760) setOpen(false);
    };
    document.addEventListener("keydown", onKey);
    window.addEventListener("resize", onResize);
    return () => {
      document.body.style.overflow = overflow;
      document.removeEventListener("keydown", onKey);
      window.removeEventListener("resize", onResize);
      menu.current?.focus();
    };
  }, [open]);
  const isResult = pathname.startsWith("/analysis/") && pathname !== "/analysis/new";
  const links = [
    { href: "/", label: "Обзор", short: "Обзор", icon: Home, active: pathname === "/" },
    {
      href: "/analysis/new",
      label: "Новый анализ",
      short: "Загрузить",
      icon: Plus,
      active: pathname === "/analysis/new",
    },
    {
      href: "/analysis",
      label: "Записи",
      short: "Записи",
      icon: Clapperboard,
      active: pathname === "/analysis" || isResult,
    },
  ];
  function toggleSidebar() {
    setCollapsed((value) => {
      const next = !value;
      try {
        localStorage.setItem("forte:sidebar-collapsed", String(next));
      } catch {
        /* Optional preference. */
      }
      return next;
    });
  }
  return (
    <div className={cn("app-shell", collapsed && "sidebar-collapsed")}>
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
        <div className="sidebar-brand-row">
          <Link
            href="/"
            className="brand"
            onClick={() => setOpen(false)}
            aria-label="Forte Vision — главная"
          >
            <img
              className="brand-symbol"
              src="/brand/forte-symbol.svg"
              alt=""
              width={36}
              height={36}
            />
            <span className="brand-name">
              forte<span>vision</span>
            </span>
          </Link>
          <button
            className="collapse-button"
            onClick={toggleSidebar}
            aria-label={collapsed ? "Развернуть навигацию" : "Свернуть навигацию"}
            aria-expanded={!collapsed}
          >
            {collapsed ? <PanelLeftOpen size={17} /> : <PanelLeftClose size={17} />}
          </button>
          <button className="mobile-close" onClick={() => setOpen(false)} aria-label="Закрыть меню">
            <X size={20} />
          </button>
        </div>
        <nav aria-label="Разделы приложения">
          {links.map(({ icon: Icon, ...item }) => (
            <Link
              key={item.href}
              href={item.href}
              className={cn("nav-link", item.active && "active")}
              title={item.label}
              aria-label={item.label}
              aria-current={item.active ? "page" : undefined}
              onClick={() => setOpen(false)}
            >
              <Icon size={18} strokeWidth={item.active ? 2 : 1.6} />
              <span>{item.label}</span>
            </Link>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <HelpDialog />
          <p className="sidebar-caption">ForteBank · Datathon 2026</p>
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
              onClick={() => setOpen(true)}
            >
              <Menu size={20} />
            </Button>
            <span className="breadcrumb-root">ForteBank</span>
            <ChevronRight size={13} className="breadcrumb-root" />
            <span>
              {isResult ? "Проверка записи" : (links.find((l) => l.active)?.label ?? "Страница")}
            </span>
          </div>
          {USE_MOCK_API && <span className="mode-badge">Деморежим</span>}
        </header>
        <main id="main-content" tabIndex={-1}>
          {children}
        </main>
        <nav className="mobile-bottom-nav" aria-label="Быстрая навигация">
          {links.map(({ icon: Icon, ...item }) => (
            <Link
              key={item.href}
              href={item.href}
              aria-label={item.label}
              aria-current={item.active ? "page" : undefined}
              className={cn(item.active && "active")}
            >
              <Icon size={19} />
              <span>{item.short}</span>
            </Link>
          ))}
        </nav>
      </div>
    </div>
  );
}
