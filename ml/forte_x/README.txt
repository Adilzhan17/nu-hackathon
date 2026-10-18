Forte X: production, single public interface

Start: https://forte.perricheno.com/analyze?pipeline=1&fresh=1
Tests: https://forte.perricheno.com/scenarios#forte-x-tests
Real tested job: /analyze?job=2c2a1c696d5b48dc8b70b2cb8397d3c4
The main / and /analysis/* routes now redirect to this single UI; see UI_HANDOFF.txt.
The separate Next shell/release_v1 is archived and its service is disabled.
Do not reactivate the retired 3000->3001 route. deploy_frontend.py guards this.
The unchanged Next API routes remain available for existing API consumers.

Forte X: VideoMAEv2 Base + human-finetuned object detector + private T4 Qwen3-VL-2B
+ guarded rules. Individual models remain selectable. The UI displays detector
checkpoint/classes, native tile boxes, window scores, timed visual observations,
evidence frames, structured memory, rule evidence, missing evidence and errors.
No organizer answer for the specific upload is passed to Qwen.

Real browser test: passed. Job2c2a1c696d5b48dc8b70b2cb8397d3c4, source15.57s,
290.99s INCLUDING waiting behind an existing user job. Six unconfirmed proposals,
12 evidence images,11 rule assessments. No confirmed violation was claimed.
Original completed test script retained at runs/forte_x/completed_test_code.cjs.
The current test entrypoint targets the canonical single UI:
  cd frontend
  timeout 390 /opt/agent-studio-runtime/node tests/forte-x-production.cjs
This starts ONE actual inference, without mocked responses. Shared T4 lock applies.
Record: runs/forte_x/production_browser_test.json. This is an integration check on
known train footage, not an independent test of accuracy.

Checks:17 frontend/API units +22 Python pipeline/job/rule checks;3 mocked browser
API-contract regressions passed before the shell retirement (logs retained).
Canonical single-UI cold-load/result checks: runs/forte_x/canonical_browser_check.json.
Initial switching availability probe:80 requests to / and /label, all HTTP200.
Annotation service PID2395625 remained unchanged. The separate UI consolidation
restarted the frontend service; source labels and manual decisions were untouched.

Production access does NOT certify detector quality. The failed newly trained v2
checkpoint was not promoted. Active detector SHA66392416b0d2..., classes cash,
document_id,key,stamp; uncalibrated score threshold0.25, native384px overlapping
tiles and class-aware NMS. Evaluation/errors are visible under /scenarios.
Detector rollback: ml/context_pipeline/publish_detector.py --rollback SAVED_REGISTRY.
Current active registry: runs/context_pipeline/detector_active.json.

Models retain their original licenses: VideoMAE NC experimental weights and YOLO
AGPL/Enterprise; public interface deployment does not grant commercial rights.
Video limit60s for full context; total job deadline300s includes queue time.
Busy/partial/failed jobs never become an automatic normal verdict.
