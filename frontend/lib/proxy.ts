import "server-only";
export async function proxy(request: Request, path: string) {
  const base = process.env.API_BASE_URL;
  const key = process.env.API_KEY;
  if (!base || !key)
    return Response.json({ error: "Analysis API is not configured" }, { status: 503 });
  if (request.method === "POST") {
    const origin = request.headers.get("origin");
    if (!origin || origin !== new URL(request.url).origin)
      return Response.json({ error: "Origin rejected" }, { status: 403 });
    if (!request.headers.get("content-type")?.startsWith("multipart/form-data;"))
      return Response.json({ error: "Multipart upload required" }, { status: 415 });
    const length = Number(request.headers.get("content-length"));
    if (length > 501 * 1024 * 1024)
      return Response.json({ error: "File too large" }, { status: 413 });
  }
  try {
    const headers = new Headers({ "X-API-Key": key, Accept: "application/json" });
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
      init.body = request.body;
      init.duplex = "half";
    }
    const upstream = await fetch(`${base.replace(/\/$/, "")}${path}`, init);
    // Never relay upstream errors, headers, or credentials to the browser.
    if (!upstream.ok)
      return Response.json(
        { error: "Analysis service rejected request" },
        { status: upstream.status },
      );
    return new Response(upstream.body, {
      status: upstream.status,
      headers: { "Content-Type": "application/json", "Cache-Control": "no-store" },
    });
  } catch {
    return Response.json({ error: "Analysis service unavailable" }, { status: 502 });
  }
}
