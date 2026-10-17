import test from "node:test";
import assert from "node:assert/strict";
import { analysisJobSchema } from "../types/analysis";
import { MAX_FILE_SIZE, validateVideo } from "../lib/utils";
const event = {
  type: "Наблюдение",
  timecode_sec: 24,
  description: "Проверить фрагмент",
  evidence_frames: [{ timestamp_sec: 24, image_base64: "/9j/4AAQ" }],
};
test("accepts all required API statuses and result types", () => {
  for (const payload of [
    { job_id: "123", status: "processing" },
    { job_id: "123", status: "failed" },
    { job_id: "123", status: "done", verdict: "violation", violations: [event, event] },
    { job_id: "123", status: "done", verdict: "no_violation", violations: [] },
  ])
    assert.equal(analysisJobSchema.safeParse(payload).success, true);
});
test("rejects contradictory verdicts and incomplete done responses", () => {
  for (const payload of [
    { job_id: "123", status: "done" },
    { job_id: "123", status: "done", verdict: "violation", violations: [] },
    { job_id: "123", status: "done", verdict: "no_violation", violations: [event] },
    {},
  ])
    assert.equal(analysisJobSchema.safeParse(payload).success, false);
});
test("rejects invalid timestamps, missing evidence and executable media URLs", () => {
  for (const bad of [
    { ...event, timecode_sec: -1 },
    { ...event, timecode_sec: Infinity },
    { ...event, evidence_frames: [] },
    { ...event, evidence_frames: Array(4).fill(event.evidence_frames[0]) },
  ])
    assert.equal(
      analysisJobSchema.safeParse({
        job_id: "123",
        status: "done",
        verdict: "violation",
        violations: [bad],
      }).success,
      false,
    );
  assert.equal(
    analysisJobSchema.safeParse({
      job_id: "123",
      status: "processing",
      preview_url: "javascript:alert(1)",
    }).success,
    false,
  );
});
test("MOV validation supports uppercase, rejects empty, oversized and wrong extensions", () => {
  assert.equal(validateVideo({ name: "cashier.MOV", size: 1024 }), null);
  for (const file of [
    { name: "cashier.mp4", size: 1024 },
    { name: "cashier.mov.exe", size: 1024 },
    { name: "cashier.mov", size: 0 },
    { name: "cashier.mov", size: MAX_FILE_SIZE + 1 },
  ])
    assert.ok(validateVideo(file));
});
