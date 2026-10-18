Terra is a temporary annotation assistant only. It is not used by Forte X video inference.
User authorized selected ambiguous frames and minimal context on 2026-10-18.
YOLO, VideoMAE and Qwen production analysis remain local.

Prepare a new read-only snapshot (output directory must not exist):
  .venv/bin/python ml/terra_review/prepare_queue.py --out runs/terra_review/NEW_RUN

Explicitly authorized bounded object pilot (at most four sequential requests):
  .venv/bin/python ml/terra_review/run_queue.py --queue runs/terra_review/NEW_RUN/queue.jsonl --external-annotation-authorized

Publish cached result links without calling a model or restarting the service:
  .venv/bin/python ml/terra_review/publish.py --results runs/terra_review/NEW_RUN/execution_v1/results.json

Pilot_v1: four completed frame requests, three approximate boxes, all class other.
No cash or ID annotations confirmed. Empty output is not a negative label.
One needs-context event at 36–48 s in Х-5 remains unresolved; this single-frame
consumer cannot establish its temporal action. It is retained in queue.jsonl.
405 unreviewed events are not automatically interpreted as ambiguous or negative.
Existing human decisions and fixed train/val split were not edited.

Results are visible at /label/objects, section "Terra · спорные кадры".
Opening the four cached links makes no provider POST. Browser checks recorded in
runs/terra_review/pilot_v1/{browser_check,public_check}.json. 27 existing assistant
and route unit tests passed. Neither these tests nor coordinate validation proves
that the proposed boxes identify the right objects; human review is required.

Before final production remove/disable external annotation initiation and server
credentials as requested, while preserving cached proposals and provenance.
This removal is a later deployment step; the current helper remains available
only in the annotation interface. Never automatically export its output as GT.
