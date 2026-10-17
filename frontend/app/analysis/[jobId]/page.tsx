import { AnalysisView } from "@/components/violations/analysis-view";
export const metadata = { title: "Проверка записи" };
export default async function JobPage({ params }: { params: Promise<{ jobId: string }> }) {
  const { jobId } = await params;
  return <AnalysisView key={jobId} id={jobId} />;
}
