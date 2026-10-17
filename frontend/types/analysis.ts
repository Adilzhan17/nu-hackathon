import { z } from "zod";
const seconds = z.number().finite().nonnegative();
export const evidenceFrameSchema = z.object({
  timestamp_sec: seconds,
  image_base64: z
    .string()
    .min(1)
    .regex(/^[A-Za-z0-9+/\r\n]+={0,2}$/),
});
export const violationSchema = z.object({
  type: z.string().min(1),
  timecode_sec: seconds,
  description: z.string().min(1),
  evidence_frames: z.array(evidenceFrameSchema).min(1).max(3),
  regulatory_source: z
    .object({
      title: z.string(),
      section: z.string().optional(),
      url: z
        .string()
        .url()
        .refine((v) => /^https:\/\//.test(v))
        .optional(),
    })
    .optional(),
});
const common = {
  job_id: z.string().min(1),
  preview_url: z
    .string()
    .optional()
    .refine((v) => !v || /^(https?:\/\/|\/(?!\/))/.test(v), "Invalid preview URL"),
};
export const analysisJobSchema = z
  .discriminatedUnion("status", [
    z.object({ ...common, status: z.literal("processing") }),
    z.object({ ...common, status: z.literal("failed"), error: z.string().optional() }),
    z.object({
      ...common,
      status: z.literal("done"),
      verdict: z.enum(["violation", "no_violation"]),
      violations: z.array(violationSchema),
    }),
  ])
  .superRefine((job, ctx) => {
    if (
      job.status === "done" &&
      ((job.verdict === "violation" && job.violations.length === 0) ||
        (job.verdict === "no_violation" && job.violations.length > 0))
    )
      ctx.addIssue({ code: "custom", message: "Verdict and violations do not match" });
  });
export type AnalysisJob = z.infer<typeof analysisJobSchema>;
export type CompletedJob = Extract<AnalysisJob, { status: "done" }>;
export type Violation = z.infer<typeof violationSchema>;
export type EvidenceFrame = z.infer<typeof evidenceFrameSchema>;
export type JobStatus = AnalysisJob["status"];
export type Verdict = CompletedJob["verdict"];
export type MockScenario = "violation" | "no_violation" | "failed";
export interface UploadMetadata {
  name: string;
  size: number;
  createdAt: number;
  mock: boolean;
  scenario?: MockScenario;
}
