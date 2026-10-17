import { LoaderCircle } from "lucide-react";
export default function Loading() {
  return (
    <div className="route-loading" role="status">
      <LoaderCircle className="spin" size={28} />
      <p>Открываем рабочее пространство…</p>
    </div>
  );
}
