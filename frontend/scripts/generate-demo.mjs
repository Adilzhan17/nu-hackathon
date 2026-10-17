// Original geometric demonstration assets. No bank footage or AI services.
// Usage: node scripts/generate-demo.mjs /absolute/path/to/ffmpeg
import sharp from "sharp";
import { mkdir, writeFile, mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { spawnSync } from "node:child_process";
const output = new URL("../public/demo/", import.meta.url);
await mkdir(output, { recursive: true });
const temp = await mkdtemp(join(tmpdir(), "forte-demo-"));
const ffmpeg = process.argv[2];
if (!ffmpeg) throw new Error("Pass an ffmpeg executable path.");
function frame(second) {
  const phase = Math.min(1, Math.max(0, (second - 20) / 12));
  const late = Math.min(1, Math.max(0, (second - 53) / 12));
  const cashX = 455 + phase * 115 + late * 125;
  const cashY = 266 + phase * 10 + late * 80;
  const time = `${String(Math.floor(second / 60)).padStart(2, "0")}:${String(Math.floor(second % 60)).padStart(2, "0")}`;
  return `<svg xmlns="http://www.w3.org/2000/svg" width="960" height="600" viewBox="0 0 960 600">
  <rect width="960" height="600" fill="#b8c3ca"/>
  <rect x="28" y="25" width="904" height="550" rx="3" fill="#d6dde0" stroke="#8c9ba5" stroke-width="2"/>
  <g stroke="#c4cdd2" stroke-width="1"><path d="M28 162H932M28 300H932M28 438H932M208 25V575M388 25V575M568 25V575M748 25V575"/></g>
  <path d="M235 25V185M235 450V575" stroke="#8e9ba4" stroke-width="17"/>
  <path d="M235 184V280M235 354V450" stroke="#a2aeb6" stroke-width="6"/>
  <path d="M235 280L306 350" stroke="#96a5af" stroke-width="3"/>
  <path d="M235 354A74 74 0 0 0 309 280" fill="none" stroke="#aab6bf" stroke-width="1" stroke-dasharray="4 4"/>
  <rect x="339" y="156" width="318" height="263" rx="8" fill="#adb5b8"/>
  <rect x="330" y="147" width="318" height="263" rx="6" fill="#d9d3c6" stroke="#b7b1a6" stroke-width="2"/>
  <rect x="352" y="171" width="104" height="67" rx="4" fill="#7f8b94"/>
  <rect x="359" y="177" width="90" height="51" rx="2" fill="#3d505d"/>
  <path d="M372 192H411M372 199H432M372 206H422M372 213H401" stroke="#7e9caa" stroke-width="2"/>
  <rect x="369" y="244" width="77" height="24" rx="2" fill="#a2abb0"/>
  <g stroke="#bdc5ca" stroke-width="1"><path d="M377 250H438M377 256H438M377 262H438"/></g>
  <rect x="364" y="319" width="76" height="57" rx="3" fill="#f1eee4" stroke="#b8b9b1"/>
  <path d="M375 333H425M375 340H418M375 347H425M375 354H405" stroke="#b6bbb6" stroke-width="2"/>
  <rect x="527" y="180" width="86" height="48" rx="3" fill="#afb7b7" stroke="#8f9b9f"/>
  <rect x="539" y="191" width="63" height="12" fill="#8e9e9e"/>
  <rect x="542" y="208" width="57" height="10" fill="#d7dacf"/>
  <path d="M490 147V410" stroke="#94a9ae" stroke-width="3"/>
  <rect x="484" y="258" width="12" height="63" fill="#d9d3c6"/>
  <g><circle cx="375" cy="475" r="28" fill="#7a8d9e" stroke="#667b8e" stroke-width="2"/><circle cx="375" cy="475" r="7" fill="#e3e9ee"/><path d="M375 462V451" stroke="#e3e9ee" stroke-width="2"/>
  <circle cx="730" cy="276" r="28" fill="#a6aca0" stroke="#8b9587" stroke-width="2"/><circle cx="730" cy="276" r="7" fill="#edf0e7"/><path d="M716 276H706" stroke="#edf0e7" stroke-width="2"/></g>
  <rect x="${cashX}" y="${cashY}" width="44" height="23" rx="2" fill="#8da18e" stroke="#597562" stroke-width="2"/>
  <path d="M${cashX + 20} ${cashY}V${cashY + 23}" stroke="#e6e5cd" stroke-width="7"/>
  <rect x="${cashX - 15}" y="${cashY - 14}" width="74" height="52" rx="4" fill="none" stroke="#aa6b43" stroke-width="2" stroke-dasharray="5 3"/>
  <g font-family="Arial, sans-serif"><text x="375" y="527" font-size="13" fill="#546979" text-anchor="middle">КАССИР</text><text x="730" y="330" font-size="13" fill="#65745e" text-anchor="middle">ПОСЕТИТЕЛЬ</text>
  <text x="${cashX - 14}" y="${cashY - 24}" font-size="11" fill="#8d633c">НАЛИЧНЫЕ · УСЛОВНО</text>
  <rect x="50" y="46" width="232" height="32" rx="3" fill="#273747"/><circle cx="65" cy="62" r="3" fill="#d4ac70"/><text x="78" y="67" font-size="11" fill="#e1e7ee" letter-spacing="1">ДЕМО / КАССОВАЯ ЗОНА</text>
  <rect x="777" y="46" width="132" height="32" rx="3" fill="#273747"/><text x="843" y="67" text-anchor="middle" font-size="12" fill="#e1e7ee" letter-spacing="1">${time} / 01:30</text>
  <text x="480" y="554" text-anchor="middle" font-size="10" letter-spacing="1" fill="#657480">УЧЕБНАЯ СХЕМА · НЕ РЕАЛЬНОЕ ВИДЕО</text></g></svg>`;
}
try {
  for (let i = 0; i < 180; i++)
    await sharp(Buffer.from(frame(i / 2)))
      .png()
      .toFile(join(temp, `${String(i).padStart(4, "0")}.png`));
  await writeFile(new URL("scene.svg", output), frame(24));
  await sharp(Buffer.from(frame(24)))
    .jpeg({ quality: 88 })
    .toFile(new URL("evidence.jpg", output).pathname);
  await sharp(Buffer.from(frame(58)))
    .jpeg({ quality: 88 })
    .toFile(new URL("evidence-58.jpg", output).pathname);
  const encoded = spawnSync(
    ffmpeg,
    [
      "-y",
      "-loglevel",
      "error",
      "-framerate",
      "2",
      "-i",
      join(temp, "%04d.png"),
      "-c:v",
      "libx264",
      "-preset",
      "fast",
      "-crf",
      "25",
      "-pix_fmt",
      "yuv420p",
      "-r",
      "24",
      "-movflags",
      "+faststart",
      new URL("cashier-preview.mp4", output).pathname,
    ],
    { stdio: "inherit" },
  );
  if (encoded.status !== 0) throw new Error("Video encoding failed");
  console.log("Generated 90-second original schematic video, JPEG evidence and SVG source.");
} finally {
  await rm(temp, { recursive: true, force: true });
}
