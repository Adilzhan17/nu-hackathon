import { Info } from "lucide-react";
import { USE_MOCK_API } from "@/services/api";

export function ModelNotice() {
  if (USE_MOCK_API) return null;
  return (
    <div className="demo-banner model-notice" role="note">
      <Info size={18} />
      <p>
        <strong>Экспериментальная модель.</strong> Типы нарушений и нормативные основания пока не
        определяются; качество смотрите в{" "}
        <a className="text-link" href="/readiness">
          аудите
        </a>
        .
      </p>
    </div>
  );
}
