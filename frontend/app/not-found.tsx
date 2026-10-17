import Link from "next/link";
import { Button } from "@/components/ui/button";
export default function NotFound() {
  return (
    <div className="page state-page">
      <h1>Страница не найдена</h1>
      <p>Проверьте адрес или вернитесь в рабочее пространство.</p>
      <Button asChild>
        <Link href="/">К обзору</Link>
      </Button>
    </div>
  );
}
