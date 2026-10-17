import { proxy } from "@/lib/proxy";
export const runtime = "nodejs";
export async function POST(request: Request) {
  return proxy(request, "/api/v1/analyze");
}
