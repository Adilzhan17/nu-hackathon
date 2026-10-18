import { proxy } from "@/lib/proxy";
import { gemmaProxy } from "@/lib/gemma-proxy";
export const runtime = "nodejs";
export async function POST(request: Request) {
  if (new URL(request.url).searchParams.get("engine") === "gemma") return gemmaProxy(request, "/jobs");
  return proxy(request, "/api/v1/analyze");
}
