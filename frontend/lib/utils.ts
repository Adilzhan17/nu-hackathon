import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}
export function timecode(seconds: number) {
  if (!Number.isFinite(seconds)) return "00:00";
  return `${Math.floor(seconds / 60)
    .toString()
    .padStart(2, "0")}:${Math.floor(seconds % 60)
    .toString()
    .padStart(2, "0")}`;
}
export function fileSize(bytes: number) {
  return `${(bytes / 1024 / 1024).toLocaleString("ru-RU", { maximumFractionDigits: 1 })} МБ`;
}
export const MAX_FILE_SIZE = 64 * 1024 * 1024;
export function validateVideo(file: Pick<File, "name" | "size">): string | null {
  if (!/\.mov$/i.test(file.name)) return "Нужен файл в формате .mov. Выберите другую видеозапись.";
  if (file.size === 0) return "Файл пустой. Выберите видеозапись с содержимым.";
  if (file.size > MAX_FILE_SIZE)
    return "Файл больше 64 МиБ — лимита сервиса. Выберите файл меньшего размера.";
  return null;
}
