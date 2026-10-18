"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";

/** The same-origin analysis workspace preserves the full, versioned job contract. */
export function ForteXWorkspace() {
  const frame = useRef<HTMLIFrameElement>(null);
  const [source, setSource] = useState("/analyze?pipeline=1&embedded=1&fresh=1");
  const [height, setHeight] = useState(1250);
  useEffect(() => {
    const job = new URLSearchParams(window.location.search).get("job");
    if (job && /^[a-f0-9]{32}$/.test(job)) setSource(`/analyze?job=${job}&embedded=1`);
    function receive(event: MessageEvent) {
      if (event.origin !== window.location.origin || event.source !== frame.current?.contentWindow) return;
      if (event.data?.type === "forte-x-height" && Number.isFinite(event.data.height)) {
        setHeight(Math.min(24000, Math.max(800, event.data.height)));
      }
      if (event.data?.type === "forte-x-job" && /^[a-f0-9]{32}$/.test(event.data.job)) {
        window.history.replaceState(null, "", `/analysis/new?job=${event.data.job}`);
      }
    }
    window.addEventListener("message", receive);
    return () => window.removeEventListener("message", receive);
  }, []);
  return <div className="page forte-x-page">
    <div className="page-heading"><div><h1>Forte X</h1>
      <p>Выберите связку или отдельные модели. Рамки, события, контекст и доказательства — в одном результате.</p>
    </div><Link className="text-link" href="/analysis/tests">Тесты Forte X →</Link></div>
    <div className="forte-x-links">
      <a href="/scenarios">CPU · RAM · GPU · качество</a>
      <Link href="/analysis/baseline">Классический X3D API</Link>
      <a href="/analyze?pipeline=1" target="_blank" rel="noopener">Открыть рабочую область отдельно</a>
    </div>
    <iframe ref={frame} title="Forte X — выбор моделей и анализ видео" src={source}
      style={{ width: "100%", height, border: 0, display: "block" }} allow="fullscreen" />
  </div>;
}
