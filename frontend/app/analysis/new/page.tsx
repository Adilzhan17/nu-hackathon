import { UploadForm } from "@/components/upload/upload-form";
import { ForteXWorkspace } from "@/components/upload/forte-x-workspace";
import { USE_MOCK_API } from "@/services/api";
export const metadata = { title: "Forte X — новый анализ" };
export default function NewAnalysisPage() {
  return USE_MOCK_API ? <UploadForm /> : <ForteXWorkspace />;
}
