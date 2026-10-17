import { proxy } from "@/lib/proxy";
export const runtime = "nodejs";
export async function GET(request: Request, { params }: { params: Promise<{ jobId: string }> }) {
  const { jobId } = await params;
  if (!/^[a-zA-Z0-9_-]{1,160}$/.test(jobId))
    return Response.json({ error: "Invalid job ID" }, { status: 400 });
  return proxy(request, `/api/v1/jobs/${encodeURIComponent(jobId)}`);
}
