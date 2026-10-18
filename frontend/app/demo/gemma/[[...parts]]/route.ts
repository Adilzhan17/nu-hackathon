import { gemmaProxy } from "@/lib/gemma-proxy";
export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export async function GET(request: Request, context: {params: Promise<{parts?: string[]}>}) {
  const { parts = [] } = await context.params;
  return gemmaProxy(request, "/" + parts.join("/"));
}
