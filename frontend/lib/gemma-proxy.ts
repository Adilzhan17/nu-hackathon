// Isolated local service. Never forwards credentials, arbitrary URLs or AI API requests.
const BASE = "http://127.0.0.1:8771";
const MAX_BYTES = 64 * 1024 * 1024;
export async function gemmaProxy(request: Request, path: string) {
  if (!/^\/(?:status|jobs(?:\/[a-f0-9]{32}(?:\/(?:media|frames\/[fd]\d{5}\.jpg))?)?)?$/.test(path))
    return Response.json({ error: "Invalid Gemma route" }, { status: 400 });
  if (!["GET", "POST"].includes(request.method)) return new Response(null, { status: 405 });
  const headers = new Headers();
  const init: RequestInit & { duplex?: "half" } = {
    method: request.method, headers, redirect: "error", cache: "no-store",
    signal: AbortSignal.any([request.signal, AbortSignal.timeout(120_000)]),
  };
  let query = "";
  if (request.method === "POST") {
    const origin = process.env.PUBLIC_ORIGIN || new URL(request.url).origin;
    if (request.headers.get("origin") !== origin)
      return Response.json({ error: "Origin rejected" }, { status: 403 });
    const length = request.headers.get("content-length") || "";
    if (!/^\d+$/.test(length) || Number(length) <= 0 || Number(length) > MAX_BYTES)
      return Response.json({ error: "Максимум 64 MB; требуется Content-Length" }, { status: 413 });
    if (path !== "/jobs") return new Response(null, { status: 405 });
    headers.set("Content-Length", length);
    headers.set("Content-Type", "application/octet-stream");
    headers.set("X-Gemma-Local-Proxy", "1");
    init.body = request.body; init.duplex = "half";
    query = "?name=" + encodeURIComponent(new URL(request.url).searchParams.get("name") || "video.mp4");
  } else if (request.headers.has("range")) headers.set("Range", request.headers.get("range")!);
  try {
    const response = await fetch(BASE + path + query, init);
    const outgoing = new Headers({"Cache-Control":"private, no-store", "X-Content-Type-Options":"nosniff"});
    for (const key of ["content-type", "content-length", "content-range", "accept-ranges"])
      if (response.headers.has(key)) outgoing.set(key, response.headers.get(key)!);
    return new Response(response.body, {status: response.status, headers: outgoing});
  } catch {
    return Response.json({error:"Отдельный сервис Gemma временно недоступен; основной анализ продолжает работать"}, {status:503});
  }
}
