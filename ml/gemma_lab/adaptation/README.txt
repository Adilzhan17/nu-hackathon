Gemma 4: local LoRA training, instructions and paired evaluation
==============================================================

Actual training commands and pinned artifact paths are in RUNBOOK.txt. train.py
uses dataset.py / PyTorch DataLoader, performs answer-only gradient updates, and
saves adapter checkpoints plus the ledger of examples actually consumed. It owns
the common GPU lock and has a hard time budget. Never run a second heavy job.
The completed experiment used 86 examples: 41 human actions and 45 weak captions,
from 10 fit source groups. Weak descriptions remain unconfirmed supervision.
The adapter is NOT automatically activated on the site. A successful gradient
update is not evidence of improved recognition; see the measured report in
runs/gemma_lab/adaptation_v1/final_report_v1/REPORT.txt when generated.

instructions_v3.txt is the corrected detailed general prompt. It adds
explicit current-image inspection, actor/cash continuity, before/after evidence,
state uncertainty and bounded JSON output. It contains no answers for individual
videos and no examples derived from reused val videos. It does not prove improved
quality until the same blind clips are processed using the old and new prompts.
Changing instructions is prompt adaptation, not fine-tuning of model weights.
instructions_v2.txt is preserved for provenance; its recipient field conflicts
with the live schema and it must not be selected for new scenario jobs.

eval.py evaluates saved outputs and never starts inference. Its input labels apply
to original whole videos. It collapses manifest windows and exact aliases, rejects
cross-split groups, and creates no window-level violation ground truth.

The active audit already froze runs/gemma_lab/adaptation_v1/frozen_group_protocol.json
(24 fit / 8 inner-train holdout / 8 reused val). Use that exact protocol through
--partition; normalization changes names only, never group membership. Do not
create another split after inspecting results. For a NEW study only, the helper
below creates a deterministic train-only partition before choosing settings:

  from ml.gemma_lab.adaptation.eval import grouped_train_partition, read_rows
  from pathlib import Path
  partition = grouped_train_partition(read_rows(Path('runs/gemma_lab/adaptation_v1/video_level_targets.jsonl')))

All training examples, instructions derived from examples, and adapter fitting must
use train_fit. Validate their group IDs with validate_fit_sources. Tune parameters
only on train_tune. Existing eight val groups are reused diagnostic data, NOT an
independent test. Do not change the frozen manifest.

Prediction JSONL record:
  {"group_id":"...","video":"...","predicted_label":null,
   "decision_scope":"whole_video","decision_basis":"insufficient_data",
   "organizer_answers_in_prompt":false,"status":"completed"}

predicted_label is 0, 1 or null. The function prediction_from_result converts a
saved Gemma result conservatively: evidence-gated supported rules can imply a
positive video; no events, normal episodes and all-insufficient rules do NOT imply
a normal video. Raw model guesses do not substitute for evidence-gated decisions.
Only an explicit completed whole-video normal assessment can emit zero.

Evaluate each condition on exactly the same scheduled source groups, with the
same frames, model revision, quantization, runtime and decoding limits:

  .venv/bin/python -m ml.gemma_lab.adaptation.eval \
    --targets runs/gemma_lab/adaptation_v1/video_level_targets.jsonl \
    --predictions runs/gemma_lab/adaptation_v1/baseline_predictions.jsonl \
    --partition runs/gemma_lab/adaptation_v1/frozen_group_protocol.json \
    --output runs/gemma_lab/adaptation_v1/baseline_metrics.json

Missing runs and insufficient_data count as abstentions, never as normal. Report:
decision coverage, accuracy among answered videos, correct/all scheduled videos,
confusion matrix INCLUDING abstentions, positive precision and end-to-end recall.
An all-abstaining system has null answered accuracy, zero decision coverage and
zero correct/all; it has NOT demonstrated 0% conventional binary accuracy because
it did not return binary decisions. Wilson intervals illustrate small-sample
uncertainty and do not make reused val an independent test.

Optional --human-events and --predicted-events compute exact-action, one-to-one
temporal IoU reference recovery, separately for every partition. Human input must
explicitly mark action_confirmed=true, eligible_for_action_training=true and
label_scope=observed_action. Proposals/rejects/states/unspecified actions are not
positives. Sparse accepted actions do not supply a complete event inventory, so
unmatched model events cannot be counted as false positives. Event precision,
full recall, event mAP and violation-event accuracy are deliberately not claimed.
Accepted intervals may preserve proposal boundaries; IoU is approximate agreement
with those intervals, not a verified boundary-quality metric.

Run checks:
  .venv/bin/python -m unittest ml.gemma_lab.adaptation.test_eval -v

Actual paired HF/NF4 inference pilot
----------------------------------
run_eval.py is a separate, bounded GPU entrypoint. It owns the common GPU flock
and refuses to load a model if another compute process is present. Do not wrap it
in another flock on the same file. A separate external timeout is appropriate:

  timeout -k 10 1850 .venv-gemma-adapt/bin/python \
    -m ml.gemma_lab.adaptation.run_eval \
    --samples runs/gemma_lab/adaptation_v1/prepared/eval_video_samples.jsonl \
    --run-root runs/gemma_lab/adaptation_v1 \
    --protocol runs/gemma_lab/adaptation_v1/frozen_group_protocol.json \
    --model artifacts/gemma_adaptation/base \
    --adapter runs/gemma_lab/adaptation_v1/train_v1/checkpoints/update_000086 \
    --output runs/gemma_lab/adaptation_v1/new_video_eval \
    --partition inner_holdout --task video --seconds 1800 \
    --conditions base_v2prompt lora_v2prompt

Model/adapter paths in the example must identify the actual pinned local base and
trained adapter. Use eval_action_samples.jsonl and --task action for the separate
action interval task. Diagnostic val is optional and separately reported; never
tune settings from it. --conditions can additionally include base_oldprompt.

This pilot samples 16 uniform frames over each video, processed as four sequential
windows of four, and two frames per accepted human action interval. It is NOT the
site's 2 fps full-sampled contextual pass and does not establish its accuracy.
Compact old/v2 prompts are frozen in eval_prompts.py. The old prompt is a compact
derivative, not the original long site prompt. Both v2 conditions receive EXACTLY
the same prior base_v2 model observations, all explicitly unconfirmed; this fixes
memory while isolating adapter weights. Adapter-specific closed-loop memory is
not evaluated here. Frame checksums, adapter checksums, prompt/config hashes and
raw outputs are saved. Decoding is greedy, capped at 128 new tokens, with 70 soft
image tokens. Backend rejects overlength inputs instead of truncating silently.

Raw whole-video classification (0/1/null) is scored separately from evidence-rule
proposals. Class 0 is a tentative classifier output, NEVER certification that all
rules are satisfied. A classifier-positive result is not promoted to a supported
violation. A supported-rule proposal needs real supplied evidence IDs and remains
unconfirmed. This separation permits measuring ordinary binary agreement without
artificially forbidding the negative class or turning guesses into site findings.

Each complete model window is saved before the next one. --resume verifies frozen
configuration/inputs and reuses completed windows. Changed adapters/prompts/data
require a new output directory. Reports include all scheduled cases (missing runs
abstain) and paired-only metrics, so a timeout cannot silently favor one condition.
Supplied action intervals support conditional top-1 action recognition; they do
not provide detection precision, full event recall, mAP or violation-event truth.
Action JSON matches training: observed_action, description, review_status=proposed.
Legacy two-field outputs are accepted and explicitly marked; any model claim of
human-confirmed review status is rejected. Both conditions use the same 128-token
generation limit, frozen before actual evaluation.

  .venv/bin/python -m unittest ml.gemma_lab.adaptation.test_run_eval -v

Additional diagnostic and full-instruction pilot
------------------------------------------------
forced_choice_eval.py compares the raw next-token logits of 0 and 1 on all 16
frozen frames jointly. This removes JSON-generation failure from a DIFFERENT
input protocol. It does not replace the sequential evaluation, establish event
evidence, or turn logits into calibrated probabilities. report.py accepts its
directory with --forced-choice-eval and preserves every primary metric.

runner.py --instructions ml/gemma_lab/adaptation/instructions_v3.txt tests the
full instruction with the existing GGUF runtime, 0.5-second overview and selected
0.25-second transition windows. compare_gguf_phone_pilot.py compares that saved
fit-video demonstration with the old pilot. One already inspected training video
is not an accuracy measurement. Instructions and trained weights are separate.
