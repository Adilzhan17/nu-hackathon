Human-reviewed incremental experiment (2026-10-17)

Website services are not restarted. Training runs on the existing remote GPU,
under the shared .gpu-work.lock. New human reviews can be submitted during a run.
An immutable snapshot means later reviews enter the next run, not a running job.

Current run: runs/human_training/batch_20261017_211756
Actions snapshot: data/human_review/exports/human_actions_20261017T211756245341Z_1214982f
Objects snapshot: data/human_review/exports/objects_20261017T211756_3b2abff2

Actions: frozen pinned SigLIP2-so400m (1,136,008,498 total parameters), plus a
trained positive text-alignment residual adapter. Six source groups supply 20
specific action intervals. Five generic 'other' actions are preserved in the
snapshot but cannot supply a consistent action-class target. Group-balanced
ridge regression uses only positive visual/text correspondences. Unmentioned
actions and intervals do NOT become negative labels. Eight original reviewed
groups reduce to six after excluding the generic action. Leave-one-source-video
out evaluation excludes all intervals from the tested video. Exact top-1/top-3
agreement with the submitted action is diagnostic, not exhaustive multi-action
accuracy and not legal-violation accuracy. Fixed ridge=1; no tuning on these
results. The final adapter includes all 20 usable intervals, but is NOT deployed.
Frame encoding uses 16 samples per labeled interval and native PTS; mean, first
half and second half features provide a basic temporal descriptor. This is not
a claim of full scenario reasoning or full backbone fine-tuning.

Objects: pretrained YOLO11m, 1280px input, fixed 50 epochs, frozen first 10
layers, AdamW. All eligible frames from a source video remain in one partition.
Held-out group chosen by deterministic hash before fitting. Last-epoch checkpoint,
never best-validation checkpoint. The library may emit automatic best-epoch
metrics; only the explicit final last.pt evaluation in report.json is authoritative.
No keys in this snapshot, so there is no evidence of key recognition quality.
The tiny diagnostic partition is not a production-quality benchmark. YOLO code
and weights use Ultralytics licensing (AGPL-3.0/enterprise), not Apache-2.0.
Official training reference: https://docs.ultralytics.com/modes/train/

Repeat on the GPU server with NEW output paths, after snapshot validation and
syncing its metadata/frames and SHA-identical source videos:
  OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    timeout 600 .venv-autolabel/bin/python -m ml.human_training.actions \
    --snapshot data/human_review/exports/NEW_ACTION_SNAPSHOT \
    --output runs/human_training/NEW_RUN/actions
  OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 timeout 900 \
    .venv-training/bin/python -m ml.human_training.detector \
    --snapshot data/human_review/exports/NEW_OBJECT_SNAPSHOT \
    --output runs/human_training/NEW_RUN/detector

Before deploying or reusing an action snapshot, run export_human_dataset's
validate_snapshot against the LIVE review store. A later correction can make an
old immutable snapshot obsolete. Do not replace original manifests or historical
checkpoints. Neither script promotes weights or starts automatic labeling.

Queue acceleration in /label: the optional checkbox hides pending suggestions
for the SAME specific action, SAME video, fully contained in a current accepted
interval. Hidden suggestions remain unreviewed; they are never exported as new
labels. Uncheck to restore them. Model confidence is not used for this decision.
Readiness API includes human_retraining status; publish.py atomically updates
only this field. Static HTML changes are served without service restarts.

Tests:
  .venv/bin/python -m unittest ml.human_training.test_training
