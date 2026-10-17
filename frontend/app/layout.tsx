import type { Metadata } from "next";
import "@fontsource-variable/inter";
import "./globals.css";
import { AppShell } from "@/components/layout/app-shell";
export const metadata: Metadata = {
  title: { default: "Forte Vision — видеоконтроль операций", template: "%s · Forte Vision" },
  description:
    "Платформа анализа видеозаписей кассовых операций. События, таймкоды и кадры для проверки специалистом.",
  icons: { icon: "/icon.svg" },
  robots: { index: false, follow: false },
};
export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="ru">
      <body>
        <AppShell>{children}</AppShell>
      </body>
    </html>
  );
}
