ForteBank: weak narration training (local only)

The experiment consumes the 99 original Russian captions, their actual video
intervals and source-level binary labels. It does not promote captions to GT.
Outputs live only under runs/narration_training; old annotations, production,
human review files and the original split manifest are read-only. A new version
may change effective whole-video membership without changing source provenance.

ACTIVE 95/5 CONFIGURATION (not trained)

prepare now consumes data/training/active_split.json by default. With split95_v1
this means 38 train videos and 2 diagnostic videos: one source-positive and one
normal. The dataset stores the diagnostic part as split=val for existing loaders,
with partition=diagnostic_test, original_split and split_policy.json preserving
its precise meaning. These two previously inspected videos are NOT independent
test data; the split cannot demonstrate 95% model accuracy.

  .venv/bin/python -m ml.narration_training.prepare \
    --split-version split95_v1 \
    --output runs/narration_training/dataset_split95_v1

Every output directory must be NEW. If dataset_split95_v1 already exists, use it
as the prepared dataset or choose a new output name; never overwrite it.
To force the old 32/8 protocol despite the active policy, use --original-split.
The two options are mutually exclusive.

All 99 existing weak pairs retain their original train-only provenance checks.
No diagnostic captions are imported. Moving six old val videos to train adds
their whole-video MIL bags and coverage windows, not new verified descriptions.
The nested 4 outer / 3 inner grouped selection still uses ONLY train groups.
Both final configurations are frozen before scoring the two diagnostic groups.

features_v1 is NOT compatible with dataset_split95_v1: clips/pairs hashes change.
Do not edit extraction.json, relabel an old cache, or reuse an old experiment as
proof of a new 95/5 run. New extraction into a fresh directory is required:

On the existing GPU server, synchronize the new narration code, ml/split95.py,
data/training/split95_v1 and the new prepared dataset first. Source originals
must match their recorded hashes. Then run (this command is not auto-started):

  OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 timeout 660 \
    .venv-autolabel/bin/python -m ml.narration_training.extract \
    --dataset runs/narration_training/dataset_split95_v1 \
    --output runs/narration_training/features_split95_v1 \
    --model artifacts/assets/quality_siglip \
    --lock /home/ubuntu/nuhack-training/.gpu-work.lock --max-seconds 600

The extractor acquires the shared lock itself. Do not wrap it in another flock.
After transferring the completed new feature directory back, run on CPU:

  OMP_NUM_THREADS=4 .venv/bin/python -m ml.narration_training.train \
    --dataset runs/narration_training/dataset_split95_v1 \
    --features runs/narration_training/features_split95_v1 \
    --output runs/narration_training/experiment_split95_v1 --epochs 60

  .venv/bin/python -m ml.narration_training.verify \
    --dataset runs/narration_training/dataset_split95_v1 \
    --features runs/narration_training/features_split95_v1 \
    --experiment runs/narration_training/experiment_split95_v1

No encoder extraction or training was launched by preparing the 95/5 version.
The old train_x3d.py controller is a separate entrypoint and is not migrated by
these commands. Human-reviewed action import remains a separate, disabled
contract; neither this split nor preparation changes that contract.

ORIGINAL 32/8 REPRODUCTION

Working commands from /root/nuhack (new output names are required):

  .venv/bin/python -m ml.narration_training.prepare \
    --original-split \
    --output runs/narration_training/dataset_v2

Already extracted features_v1 matches dataset_v1. To reproduce the CPU experiment:

  OMP_NUM_THREADS=4 .venv/bin/python -m ml.narration_training.train \
    --dataset runs/narration_training/dataset_v1 \
    --features runs/narration_training/features_v1 \
    --output runs/narration_training/experiment_v2 --epochs 60

  .venv/bin/python -m ml.narration_training.verify \
    --dataset runs/narration_training/dataset_v1 \
    --features runs/narration_training/features_v1 \
    --experiment runs/narration_training/experiment_v2

  .venv/bin/python -m ml.narration_training.report \
    --dataset runs/narration_training/dataset_v1 \
    --features runs/narration_training/features_v1 \
    --experiment runs/narration_training/experiment_v2

Load a saved checkpoint and reproduce the eight diagnostic video scores:

  .venv/bin/python -m ml.narration_training.infer \
    --checkpoint runs/narration_training/experiment_v1/narration.pt \
    --dataset runs/narration_training/dataset_v1 \
    --features runs/narration_training/features_v1 --split val \
    --output runs/narration_training/experiment_v1/reloaded_val.jsonl

Test:
  .venv/bin/python -m unittest ml.narration_training.test_narration_training -v

GPU extraction, on the existing remote machine, from
/home/ubuntu/nuhack-training. The existing autolabel environment is used read-only
(created for our earlier autolabel work); no packages or other environment change.
Only the new code and dataset directory need synchronization. Source videos and
model already exist remotely. SHA-identical video aliases are resolved explicitly.

  OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 timeout 660 \
    .venv-autolabel/bin/python -m ml.narration_training.extract \
    --dataset runs/narration_training/dataset_v1 \
    --output runs/narration_training/features_v1 \
    --model artifacts/assets/quality_siglip \
    --lock /home/ubuntu/nuhack-training/.gpu-work.lock --max-seconds 600

For a short pilot add --limit-groups 3 --max-seconds 180 and timeout 240.
The encoder verifies pinned model/file hashes and local Apache-2.0 declaration;
network access is disabled for loading. It acquires the shared flock itself,
waits at most 30 seconds and fails if busy. Do not wrap it in a second flock
on the same path (that would deadlock). Cached extraction can resume; training
always requires a fresh output directory. No new machines are provisioned.

Model: google/siglip2-so400m-patch14-384
Revision: e8e487298228002f3d8a82e0cd5c8ea9c567f57f
Parameters: 1,136,008,498 (< 3 billion), Apache-2.0.
Official model/license card:
https://huggingface.co/google/siglip2-so400m-patch14-384
Public model documentation was researched; private data was not sent anywhere.
Full checkpoint export/distribution must retain the upstream Apache notices.
Our checkpoints contain only the small adapter/head plus provenance/prototypes.

Why the original experiment used this method: only 32 train sources, with 99 weak descriptions
from 13 groups. Freezing a multilingual pretrained backbone avoids fitting a
large encoder on this tiny dataset and avoids a paid model download. Ordered
frame features feed a small temporal residual adapter. Caption cosine alignment
and positive-only action-prototype loss update the SAME adapter used for video
classification. Unit tests show that changing text changes adapter gradients.
Unmentioned actions have zero action loss, not a negative action target.
The action task is weak positive-prototype learning, not a calibrated multiclass
detector. No action accuracy is claimed without human-confirmed references.

Binary training uses all coverage windows of each source as one MIL bag; a
whole-video organizer label never becomes a target on a single window.
Baseline has exactly the same architecture, images, initialization schedule,
epoch count and parameter-selection budget; caption/action losses are disabled.
4 outer grouped folds / 3 inner grouped folds; LR selection only inside train;
fixed threshold 0.5; reused val is diagnostic, never an independent test.
The compact audit queue is automatic lexical triage, not visual adjudication.
The frozen feature cache includes val images without fitting anything to them;
no val descriptions or labels participate in fitting or parameter selection.

Human actions: 0 confirmed records in the observed export. A schema agreement
was requested. human_contract.proposed.json documents the observed version-1
export and required checks. No adapter is activated or source labels imported
before agreement; no changes to ml/human_review* or data/human_review. When the
contract is agreed, implement the read-only mapping as a separate source and
refit with identical group masks. Do not substitute weak proposals for missing
confirmed actions, or interpret rejected/absent actions as normal behavior.
