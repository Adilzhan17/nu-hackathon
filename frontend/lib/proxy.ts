import "server-only";
const MAX_MULTIPART_BYTES = 65 * 1024 * 1024;
export async function proxy(request: Request, path: string, media = false) {
  const base = process.env.API_BASE_URL;
  const key = process.env.API_KEY;
  if (!base || !key)
    return Response.json({ error: "Analysis API is not configured" }, { status: 503 });
  if (request.method === "POST") {
    const origin = request.headers.get("origin");
    const url = new URL(request.url);
    const host = request.headers.get("host") ?? url.host;
    const allowedOrigin = process.env.PUBLIC_ORIGIN || `${url.protocol}//${host}`;
    if (!origin || origin !== allowedOrigin)
      return Response.json({ error: "Origin rejected" }, { status: 403 });
    if (!request.headers.get("content-type")?.startsWith("multipart/form-data;"))
      return Response.json({ error: "Multipart upload required" }, { status: 415 });
    const lengthHeader = request.headers.get("content-length");
    if (lengthHeader !== null && !/^\d+$/.test(lengthHeader))
      return Response.json({ error: "Invalid content length" }, { status: 400 });
    if (Number(lengthHeader) > MAX_MULTIPART_BYTES)
      return Response.json({ error: "File too large" }, { status: 413 });
  }
  try {
    const headers = new Headers({
      "X-API-Key": key,
      Accept: media ? "video/mp4" : "application/json",
    });
    if (media && request.headers.has("range")) headers.set("Range", request.headers.get("range")!);
    if (request.method === "POST")
      headers.set("Content-Type", request.headers.get("content-type")!);
    const init: RequestInit & { duplex?: "half" } = {
      method: request.method,
      headers,
      cache: "no-store",
      redirect: "error",
      signal: AbortSignal.any([
        request.signal,
        AbortSignal.timeout(request.method === "POST" ? 115_000 : 18_000),
      ]),
    };
    if (request.method === "POST") {
      const length = request.headers.get("content-length");
      if (length !== null) {
        headers.set("Content-Length", length);
        init.body = request.body;
        init.duplex = "half";
      } else {
        // The Python API expects Content-Length. A browser normally provides it;
        // for chunked callers, buffer raw multipart bytes under the same limit.
        const reader = request.body?.getReader();
        const chunks: Uint8Array[] = [];
        let total = 0;
        const cancel = () => {
          if (reader) void reader.cancel().catch(() => {});
        };
        init.signal!.addEventListener("abort", cancel, { once: true });
        try {
          if (reader) {
            while (!init.signal!.aborted) {
              const chunk = await reader.read();
              if (chunk.done) break;
              total += chunk.value.byteLength;
              if (total > MAX_MULTIPART_BYTES) {
                cancel();
                return Response.json({ error: "File too large" }, { status: 413 });
              }
              chunks.push(chunk.value);
            }
          }
          init.signal!.throwIfAborted();
        } finally {
          init.signal!.removeEventListener("abort", cancel);
          reader?.releaseLock();
        }
        const bytes = new Uint8Array(total);
        let offset = 0;
        for (const chunk of chunks) {
          bytes.set(chunk, offset);
          offset += chunk.byteLength;
        }
        headers.set("Content-Length", String(total));
        init.body = bytes;
      }
    }
    const upstream = await fetch(`${base.replace(/\/$/, "")}${path}`, init);
    // Never relay upstream errors, headers, or credentials to the browser.
    if (!upstream.ok)
      return Response.json(
        { error: "Analysis service rejected request" },
        { status: upstream.status },
      );
    if (media) {
      const responseHeaders = new Headers({
        "Content-Type": "video/mp4",
        "Cache-Control": "private, no-store",
      });
      for (const name of ["Content-Length", "Content-Range", "Accept-Ranges"])
        if (upstream.headers.has(name)) responseHeaders.set(name, upstream.headers.get(name)!);
      return new Response(upstream.body, { status: upstream.status, headers: responseHeaders });
    }
    const payload = await upstream.json();
    // Only this job's preview can be fetched with server credentials. Never turn
    // arbitrary media URLs from an upstream response into an authenticated proxy.
    if (request.method === "GET" && payload && typeof payload === "object" && payload.preview_url) {
      const preview = new URL(payload.preview_url, base);
      const expectedPath = `${path}/preview`;
      if (
        preview.origin !== new URL(base).origin ||
        preview.pathname !== expectedPath ||
        preview.search ||
        preview.hash
      )
        delete payload.preview_url;
      else payload.preview_url = expectedPath.replace("/api/v1/", "/api/");
    }
    return Response.json(payload, {
      status: upstream.status,
      headers: { "Content-Type": "application/json", "Cache-Control": "no-store" },
    });
  } catch {
    return Response.json({ error: "Analysis service unavailable" }, { status: 502 });
  }
}
