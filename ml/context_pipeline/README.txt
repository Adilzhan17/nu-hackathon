Forte: experimental detector and context pipeline

Live pages (served by the existing annotation process, no restart):
  /training                 resource panel and training results
  /scenarios                CPU/RAM/T4, training epochs, real detector comparisons
  /analyze?pipeline=1        explicit VideoMAE + YOLO11 -> private T4 Qwen experiment
  /analyze                  selecting YOLO8/YOLO11 also shows our target detector boxes

The exact selection {videomaev2,yolo11} activates the contextual experiment in
the per-upload infer_video subprocess. Other selections retain their classifier
workflow; their object overlay reads the same detector_active registry.
This document describes the older detector experiment. The current private jury
API uses the local Gemma scenario pipeline. External annotation initiation and
credentials were removed on 2026-10-18, including the obsolete queue/publisher
scripts. Historical annotation provenance is preserved; manual boxes remain
human records and are not overwritten by inference.

Data:
  detector_dataset_v3: read-only SQLite snapshot, only complete/certain human
  decisions inside the ORIGINAL frozen train split. One prior diagnostic source
  group stays held out, including every new frame from that source. The previous
  checkpoint did not train on this held-out group. No fixed val video is used.
  Human originals, native crops and photometric synthetic frames have separate
  provenance. Synthetic variants preserve geometry, perspective and original
  hand occlusion. Crop labels are clipped; crops retaining <50% of another target
  are rejected. All other visible labelled targets remain labelled.
  Earlier detector_dataset_v2 copy-paste candidates are preserved but NOT used.

  42 eligible human frames (4 uncertain frames excluded): 28 train, 14 diagnostic.
  Training: 28 originals + 45 crops + 24 photometric variants = 97 images.
  Diagnostic: 14 unaugmented real frames from one source video, not independent test.

Train commands:
  .venv/bin/python -m ml.context_pipeline.prepare_detector \
    --output runs/context_pipeline/detector_dataset_UNIQUE
  # T4 only, own .venv-detector; GPU lock is acquired INSIDE the trainer.
  env YOLO_CONFIG_DIR=/home/ubuntu/nuhack-training/runs/context_pipeline/yolo_config \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 \
    timeout 960 .venv-detector/bin/python -m ml.context_pipeline.train_detector \
      --dataset runs/context_pipeline/detector_dataset_v3 \
      --weights runs/context_pipeline/previous_detector.pt \
      --output runs/context_pipeline/detector_training_UNIQUE --epochs 20 --max-seconds 900
  No checkpoint/threshold is selected on diagnostic scores. Fixed last epoch.

Matched CPU evaluation, run once for each checkpoint:
  .venv/bin/python -m ml.context_pipeline.evaluate_detector \
    --dataset runs/context_pipeline/detector_dataset_v3 \
    --weights CHECKPOINT --name old|new \
    --output runs/context_pipeline/detector_evaluation_v3
  .venv/bin/python -m ml.context_pipeline.review_examples \
    runs/context_pipeline/detector_evaluation_v3
  P/R use confidence=0.25, IoU=0.5. AP is integrated across scores >=0.001.
  Full frame versus overlapping native 384px tiles (25% overlap), model input
  960px, class-aware NMS=0.5. Original videos are only 464x848; enlargement cannot
  recover missing pixels. No hand/table subgroup metrics without human tags.

Publication and rollback:
  .venv/bin/python -m ml.context_pipeline.publish_detector --checkpoint PATH \
    --evaluation runs/context_pipeline/detector_evaluation_v3/report.json \
    --name 'YOLO11m human v2'
  .venv/bin/python -m ml.context_pipeline.publish_detector --rollback SAVED_REGISTRY
  Every new upload reads registry afresh; old job results retain their original
  version and evidence. detector_active.json contains the rollback registry path.

Monitoring:
  ml/live_console/collect.py performs bounded read-only SSH sampling on port15171,
  and atomically publishes status through the already-served /scenarios HTML.
  nuhack-live-status.timer invokes a <=20 second oneshot every15 seconds.
  No background GPU loop. Stale/unavailable metrics are shown explicitly.
  Annotation service and Next.js were not restarted.

Context uploads: <=60s; original global upload/queue/inference deadline300s.
CPU VideoMAE candidates and target boxes -> private T4 Qwen, FP16 efficient SDPA.
One shared GPU lease, bounded external timeout, no cloud AI, no audio/identities.
Busy GPU returns an explicit failure, never a normal/negative verdict.
Objects/actions/rules/memory remain separate unconfirmed layers. Missing actions
in sampled frames do not prove absence. ID candidate does not establish identity.

VideoMAE action-head experiment:
  train_actions.py consumes only confirmed train intervals; Dataset/DataLoader,
  frozen VideoMAE features, 80 fixed epochs, group-held-out baseline comparison.
  action_training_v1: 20 intervals/6 groups; nearest-neighbor baseline25%, learned
  head20% annotation agreement. Ten held-out examples have classes absent from
  their training fold. No demonstrated improvement; head NOT activated.

Licenses: Qwen3-VL-2B Apache2; selected VideoMAEv2-Base weights CC-BY-NC4;
Ultralytics detector AGPL3/Enterprise. This remains an experimental interface.

Actual matched result (2026-10-18; fixed confidence 0.25):
  old native tiles: cash P=0 R=0 AP50=.127 mAP50-95=.043;
                    document_id P=1 R=.083 AP50=.510 mAP50-95=.182.
  new native tiles: cash P=0 R=0 AP50=.098 mAP50-95=.024;
                    document_id P=.214 R=.25 AP50=.176 mAP50-95=.066.
  New detector completed 20 epochs in 607.7s, SHA
  d20cc1fb421953ab465a0cb65eaa4243ccd509873b8c94826ccd824c4bac709c.
  It was NOT promoted: cash still fails and document precision/AP regress.
  No reliable cash detector has been demonstrated on held-out videos.
  Previous human-finetuned v1 remains active, now with native tiles and explicit
  provenance/limitations in every YOLO overlay. Do not describe this as a quality
  improvement from retraining. See deployment_decision.json and evaluation_v3.

Verified public-site pilots:
  /analyze?job=788fb3b1b8a0439387d0c51ee17fd565
  Full hybrid: completed, 15.57s source, 159.4s upload-to-result; VideoMAE11.1s,
  detector27.4s, Qwen88.4s. 16 survey frames plus 18/24 refinement frames.
  Six schema-valid proposals (overlapping windows, NOT six distinct confirmed
  events); six state proposals rejected for absent evidence-frame references.
  Eleven rule verdicts insufficient_data. No confirmed violation is claimed.
  Twelve evidence images and real detector SVG boxes verified in the browser.
  This known train-source pilot validates integration, not generalization.
  /analyze?job=5310cbbd556b4c39a3e2ea778ed0737b
  Legacy YOLO8 + YOLO11 classifier modes completed, both target-object overlays
  identify the same actual human-finetuned checkpoint and taxonomy.
  Unit checks: 22 passed. Protected source split and 111 image/label hashes
  checked; diagnostic synthetic frames=0; source-group overlap=0.

Publication now refuses zero true positives for cash/document_id at the fixed
threshold, or a per-class precision/recall/mAP50-95 regression. This is a minimal
regression guard, not certification from one diagnostic video. The current v2
candidate fails this guard and is retained only as a training/evaluation artifact.
Four excluded uncertain TRAIN images are linked from /scenarios for review;
hand/table questions for diagnostic examples remain separate and non-training.

Organizer continuation finished: 22/22 complete; report_final22 verified 9,886
protected files unchanged. 307 visual proposals, 21 conflicts, compact queue9;
206 invalid model proposals excluded. 17 weak train scenarios and20 human
action intervals remain separate. All242 guarded rule assessments are
insufficient_data; no automatic strong target or legal violation was created.
Context ablation:3/11 references recovered with history vs2/11 without, on16
matched windows; this small diagnostic does not establish general improvement.
All owned GPU jobs ended. Main annotator PID2395625 remained running throughout.
Precision is undefined when no prediction exists (shown as dash on the site);
numeric files retain the explicit zero-division convention with a boolean flag.
