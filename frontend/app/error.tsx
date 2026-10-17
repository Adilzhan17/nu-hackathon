"use client";
import { Button } from "@/components/ui/button";
export default function ErrorPage({ reset }: { reset: () => void }) {
  return (
    <div className="page state-page">
      <h1>Не удалось открыть страницу</h1>
      <p>Попробуйте загрузить её ещё раз.</p>
      <Button onClick={reset}>Повторить</Button>
    </div>
  );
}
