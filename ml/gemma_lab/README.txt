GEMMA 4 — COMPATIBILITY AND HISTORICAL OPERATIONS

The canonical runtime now lives in forte_vision/. Start with README.txt and
docs/ARCHITECTURE.txt at the repository root. Modules here forward to the new
implementation; adaptation/ retains frozen research inputs. UI ownership moved
to frontend/analysis/index.html. Existing job and model directories are unchanged.

Public MVP: https://forte.perricheno.com/ (redirects to /demo/gemma)
Code: ml/gemma_lab/; jobs/results: runs/gemma_lab/uploads/<uuid>/.
GPU assets: /home/ubuntu/nuhack-training/artifacts/gemma_lab/.
GPU Python: .venv-gemma-lab/bin/python; no changes to other virtual environments.

Model: google/gemma-4-12B-it-qat-q4_0-gguf
Revision: 29d097773436b69ff9feafd636ab4cf873786537
License: Apache-2.0 (official Google model card preserved beside weights).
Runtime: llama.cpp b11716, CUDA12.8. Weight and archive SHA256 values in bootstrap.py.
Private Ubuntu ABI libraries are unpacked in private-sysroot; no host packages replaced.
Links for provenance:
https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-gguf
https://github.com/ggml-org/llama.cpp/releases/tag/b11716

FLOW
Upload MP4/MOV <=180sec, <=64MiB. Decode only video. Main host PyAV retains original
frame dimensions; overview requests all frames sampled at .5sec. Chronological
windows contain 12 frames with 2-frame overlaps. Up to two selected transitions
are reviewed at .25sec with before/after context. All JPEGs remain on our hosts.
The vision model internally resizes/tokenizes images; original resolution alone
is not a guarantee that tiny cash or ID is legible. Vision token budget is 512.

The production web flow selects adaptation/instructions_v4.txt. The prompt contains
general rules, current timestamped frames and structured
fallible memory. No original filename, organizer answer, transcript, human label,
train/val target or audio is sent to the model. Context is built from earlier
visual proposals only, with uncertainty and last-known/current-unknown state.
Raw responses, rejected observations, frame hashes, code hashes, prompt hash,
model revision and per-window completion/usage information are preserved.

Every result is PROPOSED, never human-confirmed or ground truth. Nothing is added
to training data, human-review files or production model checkpoints. Red rule
cards need structured evidence gates; orange cards show unconfirmed Gemma claims.
The inherited rule engine supports only a subset of positive scenarios: many
rules intentionally remain insufficient_data. Raw model claims are displayed
separately, including their timestamps, even when they fail the evidence gate.
Sampling cannot prove a missing action; intervals are evidence spans, approximate
boundaries. The run is not an accuracy benchmark or an independent test result.
No audio, identity recognition, document transcription or fabricated boxes.

GPU SAFETY
All runs acquire /home/ubuntu/nuhack-training/.gpu-work.lock. A busy lock or any
existing GPU compute process defers the request; no other process is stopped.
Only our own child llama-server is stopped on completion/error. No persistent
GPU server. Default job limit900sec, outer timeout925sec with kill grace10sec.
One web job at a time. Results sync to main host approximately every8sec.
HTTP /status busy reflects web queue, not an external GPU reservation.

RULE SOURCES
Verified public metadata is attached by regulatory.py. The UI shows the title,
section and official Adilet URL only for mapped rules, with a limitation note.
In particular, keys left on a safe during an employee departure map to section80
as a review signal: that section delegates the concrete custody procedure to the
bank's internal documents, so the service does not claim a final legal violation.

BOOTSTRAP ON EXISTING GPU (downloads weights only, no AI API)
python3 -m venv .venv-gemma-lab
.venv-gemma-lab/bin/python -m ml.gemma_lab.bootstrap --root "$PWD"
.venv-gemma-lab/bin/python -m ml.gemma_lab.portable_runtime --root "$PWD"
.venv-gemma-lab/bin/python -m ml.gemma_lab.runner --job runs/gemma_lab/<prepared-id> --seconds 900

MAIN HOST
.venv/bin/python -m ml.gemma_lab.server --port 8771
The independent loopback service is reached through the same-origin Next routes:
GET /demo/gemma/..., POST /api/analysis?engine=gemma&name=... . Existing analysis
POST requests without engine=gemma keep their original handler.

JURY API PROFILE
Set NUHACK_JURY_ENGINE=gemma on the process serving /api/v1. Authenticated MOV
uploads are streamed to the same loopback worker and use a bounded 270-second
profile with at most18 evenly scheduled overview frames. Completed evidence-gated rules are converted to the
required binary verdict, timestamp and 1–3 base64 JPEG frames. Partial runs fail
instead of being reported as no_violation. Legacy X3D job IDs remain readable.

DEPLOYMENT / ROLLBACK
nuhack-gemma-lab.service owns CPU service8771.
nuhack-gemma-web.service serves the separate .next-gemma-lab build on3002.
nuhack-gemma-route.service adds exactly one tagged loopback NAT rule3000->3002.
The original annotator8765 and frontend3000 remain alive. Rollback:
systemctl stop nuhack-gemma-route
This removes only the rule tagged gemma-lab-web-v1. Original frontend resumes;
remove the Gemma link from ml/analyze.html if withdrawing the tab. Do not stop
Gemma during an active job; its remote task is bounded even if transport fails.

CHECKS
.venv/bin/python -m unittest ml.gemma_lab.test_protocol ml.gemma_lab.test_bridge -v
cd frontend && /opt/agent-studio-runtime/node --import tsx --test tests/gemma-proxy.test.ts
The browser pilot and factual pilot summary are saved under runs/gemma_lab/.

DURATION FIX
Maximum video duration180sec now includes all original project clips (longest
142.53sec). Metadata is checked in the browser where supported, and again on the
server before HTTP202/GPU acceptance. Decoded timestamps are bounded separately.
A failed validation has no inference results, violation counts or empty player.
The900sec GPU budget is unchanged: accepting180sec does not guarantee completion
within that budget; partial coverage remains explicit. The actual Х-4 regression
uses72.7sec /146 overview frames, with original failures retained unchanged.
Browser regression (mock API, no GPU):
/opt/agent-studio-runtime/node ml/gemma_lab/test_ui.cjs

RESPONSE FORMAT FIX
A real72.7sec upload exposed a truncated1600-token answer and an input-memory
schema echoed as output. The runner now requires events/states/rule_checks arrays
and constrains generation using the pinned llama.cpp response_format.schema API.
Per-window output is bounded to3 events,3 states,2 rule checks and3 evidence IDs
per item; max2200 tokens is a ceiling, not a target. Empty observations remain
possible and do not prove absence. All original raw failures are preserved.
Reference: https://github.com/ggml-org/llama.cpp/blob/b11716/tools/server/README.md
Initial empty rule templates are no longer shown. The UI separates response
coverage from observations and hides verdict counters when no response exists.
