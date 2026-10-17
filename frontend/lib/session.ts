import type { UploadMetadata } from "@/types/analysis";
const urls = new Map<string, string>();
const metadata = new Map<string, UploadMetadata>();
export function rememberUpload(id: string, file: File, meta: UploadMetadata) {
  const old = urls.get(id);
  if (old) URL.revokeObjectURL(old);
  urls.set(id, URL.createObjectURL(file));
  metadata.set(id, meta);
  try {
    sessionStorage.setItem(`forte:${id}`, JSON.stringify(meta));
    sessionStorage.setItem("forte:last-job", id);
  } catch {
    /* Memory fallback works when storage is unavailable. */
  }
}
export function getVideoUrl(id: string) {
  return urls.get(id);
}
export function getMetadata(id: string): UploadMetadata | undefined {
  if (metadata.has(id)) return metadata.get(id);
  try {
    const value = sessionStorage.getItem(`forte:${id}`);
    if (value) return JSON.parse(value);
  } catch {
    /* A reload may need the source file again. */
  }
}
export function releaseVideo(id: string) {
  const url = urls.get(id);
  if (url) URL.revokeObjectURL(url);
  urls.delete(id);
}
