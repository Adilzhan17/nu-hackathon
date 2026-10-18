"""Bounded, resume-safe local Gemma comparison. Acquires the shared GPU lock.

Three conditions: compact old-style prompt, compact v2 prompt, same v2 + trained
LoRA. Same HF NF4 base, images and greedy decoding. This 16-frame/2-frame pilot is
NOT production 2 fps contextual video evaluation. No target text enters prompts.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import time

from .eval import classify, digest, normalize_partition, read_rows
from .eval_prompts import OLD, V2, VIDEO_CONTRACT, ACTION_CONTRACT, prompt
from .taxonomy import ACTION_CODES, TAXONOMY_VERSION, TAXONOMY_SOURCE_SHA256

CONDITIONS = ("base_oldprompt", "base_v2prompt", "lora_v2prompt")
MAX_NEW_TOKENS = 128


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def sha_text(value):
    return hashlib.sha256(value.encode()).hexdigest()


def adapter_digest(path):
    path = Path(path)
    expected = [path / "adapter_config.json", path / "adapter_model.safetensors"]
    if not all(item.is_file() for item in expected):
        raise ValueError("Expected a saved adapter config and safetensors checkpoint")
    return {item.name: digest(item) for item in expected}


def blind_sample(row, run_root):
    """Strict projection drops targets, notes and organizer fields before inference."""
    task = row["task"]
    if task not in {"video", "action"}:
        raise ValueError("Unknown sample task")
    refs = row["frames"]
    if len(refs) != (16 if task == "video" else 2):
        raise ValueError("Reduced protocol requires exactly 16 video or two action frames")
    frames = []
    last = -1.0
    for index, frame in enumerate(refs):
        timestamp = float(frame["timestamp"])
        if not math.isfinite(timestamp) or timestamp < 0 or timestamp <= last:
            raise ValueError("Frame times must be finite and strictly chronological")
        last = timestamp
        path = (run_root / frame["path"]).resolve()
        if not path.is_relative_to(run_root.resolve()):
            raise ValueError("Frame path escapes the frozen run root")
        if digest(path) != frame["sha256"]:
            raise ValueError("Frame checksum mismatch")
        frames.append(dict(id=f"f{index}", path=str(path), timestamp=timestamp, sha256=frame["sha256"]))
    return dict(sample_id=row["sample_id"], group_id=row["group_id"], task=task, frames=frames)


def parse_answer(raw, task, frame_ids):
    """Strict JSON; malformed generations abstain, never guessed/repaired labels."""
    text = raw.strip()
    fenced = re.fullmatch(r"```(?:json)?[ \t]*\r?\n([\s\S]*?)\r?\n```", text)
    wrapper_removed = bool(fenced)
    if fenced:
        text = fenced.group(1)
    try:
        value = json.loads(text)
        if not isinstance(value, dict):
            raise ValueError("Expected a JSON object")
        if task == "video":
            if set(value) != {"predicted_label", "rule_verdict", "evidence_frames", "reason"}:
                raise ValueError("Unexpected video output schema")
            label = value["predicted_label"]
            if label is not None and (type(label) is not int or label not in (0, 1)):
                raise ValueError("Label must be 0, 1 or null")
            verdict = value["rule_verdict"]
            if verdict not in {"violation_supported", "insufficient_data"}:
                raise ValueError("Unknown evidence verdict")
            refs = value["evidence_frames"]
            allowed = set(frame_ids)
            if not isinstance(refs, list) or not all(isinstance(ref, str) and ref in allowed for ref in refs):
                raise ValueError("Invalid frame evidence")
            if not isinstance(value["reason"], str):
                raise ValueError("Reason must be text")
            status = "accepted"
            if verdict == "violation_supported" and len(set(refs)) < 2:
                verdict, status = "insufficient_data", "positive_rule_without_two_frame_evidence"
            return dict(predicted_label=label, raw_predicted_label=value["predicted_label"],
                        rule_verdict=verdict, raw_rule_verdict=value["rule_verdict"],
                        evidence_frames=refs, reason=value["reason"], parse_status=status,
                        wrapper_removed=wrapper_removed)
        if set(value) not in ({"observed_action", "description"},
                              {"observed_action", "description", "review_status"}):
            raise ValueError("Unexpected action output schema")
        if "review_status" in value and value["review_status"] != "proposed":
            raise ValueError("Model action cannot claim human confirmation")
        action = value["observed_action"]
        if action not in set(ACTION_CODES) | {"unknown"} or not isinstance(value["description"], str):
            raise ValueError("Unknown action code or invalid description")
        return dict(observed_action=action, description=value["description"], review_status="proposed",
                    parse_status="accepted", legacy_two_field_output="review_status" not in value,
                    wrapper_removed=wrapper_removed)
    except (ValueError, TypeError, KeyError) as error:
        return dict(predicted_label=None, observed_action="unknown", parse_status="invalid_json_or_schema",
                    parse_error=str(error), wrapper_removed=wrapper_removed)


def action_metrics(samples, rows):
    lookup = {row["sample_id"]: row for row in rows}
    counts = Counter()
    per_action = {}
    for sample in samples:
        ref = sample["reference"]
        action = ref["observed_action"]
        if ref.get("action_confirmed") is not True:
            raise ValueError("Action accuracy requires explicit human confirmation")
        if action in {"other_observed_action", "unclear_action", "cash_visible", "safe_open_visible"}:
            counts["excluded_unspecified_or_state"] += 1
            continue
        pred = lookup.get(sample["sample_id"], {}).get("observed_action", "unknown")
        answered = pred not in {"unknown", "unclear_action"}
        counts["scorable"] += 1
        counts["answered"] += int(answered)
        counts["correct"] += int(pred == action)
        item = per_action.setdefault(action, dict(references=0, answered=0, correct=0))
        item["references"] += 1
        item["answered"] += int(answered)
        item["correct"] += int(pred == action)
    total = counts["scorable"]
    answered = counts["answered"]
    return dict(**dict(counts), top1_correct_fraction_all=counts["correct"] / total if total else None,
                accuracy_on_answered=counts["correct"] / answered if answered else None,
                decision_coverage=answered / total if total else None, per_action=per_action,
                task="Conditional action classification on supplied human intervals, not event detection",
                event_precision=None, event_recall=None, event_mAP=None,
                limitation="Sparse accepted intervals; boundaries may not be independently retimed.")


def metrics(samples, rows, conditions=CONDITIONS, include_paired=True):
    reports = {}
    for condition in conditions:
        selected = [row for row in rows if row["condition"] == condition]
        sections = {}
        for part in sorted({row["partition"] for row in samples}):
            videos = [row for row in samples if row["task"] == "video" and row["partition"] == part]
            actions = [row for row in samples if row["task"] == "action" and row["partition"] == part]
            ids = {row["sample_id"] for row in videos + actions}
            relevant = [row for row in selected if row["sample_id"] in ids]
            targets = [dict(video=row["video"], group_id=row["group_id"],
                            split="val" if part == "diagnostic_val" else "train",
                            label=row["reference"]["label"], label_scope="whole_video") for row in videos]
            sections[part] = dict(
                whole_video=classify(targets, [row for row in relevant if row["task"] == "video"]),
                rule_supported_whole_video=classify(targets, [dict(row,
                    predicted_label=1 if row.get("rule_verdict") == "violation_supported" else None,
                    decision_basis="unconfirmed_rule_proposal_with_minimum_evidence_ids")
                    for row in relevant if row["task"] == "video"]),
                supplied_action_intervals=action_metrics(actions, [row for row in relevant if row["task"] == "action"]))
        reports[condition] = sections
    by_sample = Counter(row["sample_id"] for row in rows)
    paired_ids = {sample_id for sample_id, count in by_sample.items() if count == len(conditions)}
    result = dict(conditions=reports, fully_paired_sample_ids=sorted(paired_ids),
                fully_paired_samples=len(paired_ids), scheduled_samples=len(samples),
                same_frames_per_condition=True, same_greedy_decoding=True,
                model_quantization="HF NF4, not deployed GGUF QAT Q4_0",
                protocol="Reduced pilot: 16 uniform frames per whole video in 4 chronological windows with fallible memory; 2 per human interval",
                paired_memory="Both v2 conditions receive identical earlier base_v2 visual proposals; no labels, no adapter-specific history",
                old_prompt_is_compact_derivative_not_site_prompt=True,
                validation_set_is_independent_test=False, no_threshold_selection=True,
                video_classification_is_unconfirmed=True,
                zero_class_prediction_is_not_normal_clearance=True,
                supported_rule_proposals_are_not_verified_violations=True,
                causal_comparison="base_v2prompt versus lora_v2prompt isolates adapter weights on identical prompts",
                source_answers_in_model_prompts=False)
    if include_paired:
        result["paired_only_conditions"] = metrics(
            [sample for sample in samples if sample["sample_id"] in paired_ids],
            [row for row in rows if row["sample_id"] in paired_ids], conditions,
            include_paired=False)["conditions"]
    return result


def run(args):
    from .hf_backend import MODEL_ID, MODEL_REVISION, MAX_SOFT_TOKENS, load_backend
    run_root = args.run_root.resolve()
    output = args.output.resolve()
    if not output.is_relative_to(run_root):
        raise ValueError("Output must remain inside the isolated adaptation run")
    if not 60 <= args.seconds <= 1800:
        raise ValueError("Evaluation must have a 60–1800 second budget")
    protocol = json.loads(args.protocol.read_text())
    assignment = normalize_partition(protocol)["assignments"]
    samples = []
    for manifest in args.samples:
        samples.extend(read_rows(manifest))
    if args.task:
        samples = [row for row in samples if row["task"] == args.task]
    if args.partition:
        samples = [row for row in samples if row["partition"] == args.partition]
    if not args.conditions or len(set(args.conditions)) != len(args.conditions):
        raise ValueError("Conditions must be nonempty and unique")
    conditions = tuple(condition for condition in CONDITIONS if condition in args.conditions)
    if "lora_v2prompt" in conditions and "base_v2prompt" not in conditions:
        raise ValueError("Paired LoRA evaluation requires base_v2prompt for identical baseline visual memory")
    if not samples:
        raise ValueError("No samples match the selected task/partition")
    if len({row["sample_id"] for row in samples}) != len(samples):
        raise ValueError("Duplicate sample IDs")
    for row in samples:
        expected = "inner_train_holdout" if row["partition"] == "inner_holdout" else "reused_val_diagnostic" if row["partition"] == "diagnostic_val" else None
        if expected is None or assignment.get(row["group_id"]) != expected:
            raise ValueError("Evaluation sample outside its frozen held-out partition")
    blind = {row["sample_id"]: blind_sample(row, run_root) for row in samples}
    config = dict(schema_version="gemma_reduced_eval_v1", model_id=MODEL_ID, model_revision=MODEL_REVISION,
                  evaluator_code_sha256={name: digest(Path(__file__).parent / name) for name in
                                         ("run_eval.py", "eval.py", "eval_prompts.py", "taxonomy.py", "hf_backend.py")},
                  manifests={str(path.resolve().relative_to(run_root)): digest(path) for path in args.samples},
                  protocol_sha256=digest(args.protocol), adapter_checksums=adapter_digest(args.adapter),
                  prompt_hashes={"old": sha_text(OLD), "v2": sha_text(V2),
                                 "video_contract": sha_text(VIDEO_CONTRACT), "action_contract": sha_text(ACTION_CONTRACT)},
                  action_codes=list(ACTION_CODES), conditions=list(conditions),
                  taxonomy_version=TAXONOMY_VERSION, taxonomy_source_sha256=TAXONOMY_SOURCE_SHA256,
                  max_new_tokens=MAX_NEW_TOKENS, max_soft_tokens_per_image=MAX_SOFT_TOKENS,
                  task_filter=args.task, partition_filter=args.partition,
                  sample_ids=[row["sample_id"] for row in samples],
                  protocol="16 frames in 4 sequential windows, last-known model memory; action 2 frames",
                  paired_memory="same baseline_v2 visual history for both v2 conditions")
    if output.exists() and not args.resume:
        raise FileExistsError("Use a new output or explicit --resume")
    output.mkdir(parents=True, exist_ok=True)
    frozen = output / "config.json"
    if frozen.exists():
        if json.loads(frozen.read_text()) != config:
            raise ValueError("Resume rejected: model, adapter, manifests, prompts or protocol changed")
    else:
        write_json(frozen, config)
    records_path = output / "predictions.jsonl"
    rows = read_rows(records_path) if records_path.exists() else []
    done = {(row["sample_id"], row["condition"]) for row in rows}
    if len(done) != len(rows) or any(sample not in blind or condition not in conditions for sample, condition in done):
        raise ValueError("Invalid or duplicate saved evaluation records")
    start = time.monotonic()
    status = dict(status="waiting_gpu", started_at=time.time(), elapsed_sec=0,
                  scheduled=len(samples) * len(conditions), completed=len(rows), max_seconds=args.seconds)
    backend = None

    def save():
        status.update(elapsed_sec=round(time.monotonic() - start, 3), completed=len(rows))
        write_json(output / "status.json", status)
        write_json(output / "metrics.json", metrics(samples, rows, conditions))

    def interrupted(*_):
        raise TimeoutError("Bounded evaluation deadline/interruption; completed predictions preserved")

    if len(done) == len(samples) * len(conditions):
        status["status"] = "completed"
        save()
        return status
    old_term, old_alarm = signal.signal(signal.SIGTERM, interrupted), signal.signal(signal.SIGALRM, interrupted)
    signal.setitimer(signal.ITIMER_REAL, args.seconds)
    save()
    try:
        with args.lock.open("a") as lease:
            try:
                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("Shared GPU lock is busy; no model loaded") from error
            foreign = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True).strip()
            if foreign:
                raise RuntimeError("GPU has active compute processes; no model loaded")
            status["status"] = "loading_model"
            save()
            backend = load_backend(args.model, training=False, adapter_path=args.adapter)
            write_json(output / "runtime.json", backend.provenance())
            from PIL import Image
            # Interleave all conditions per sample, rather than finishing one
            # entire condition first and leaving unpaired comparisons at timeout.
            for sample in samples:
                item = blind[sample["sample_id"]]
                for condition in conditions:
                    if (item["sample_id"], condition) in done:
                        continue
                    if args.seconds - (time.monotonic() - start) < 20:
                        raise TimeoutError("Budget exhausted before next prediction")
                    chunks = [item["frames"][offset:offset + 4] for offset in range(0, len(item["frames"]), 4)]
                    window_results = []
                    memory = []
                    before_sample = time.monotonic()
                    for window_index, frames in enumerate(chunks):
                        if args.seconds - (time.monotonic() - start) < 20:
                            raise TimeoutError("Budget exhausted before next window")
                        status.update(status="processing", sample_id=item["sample_id"], condition=condition,
                                      window=window_index, windows_per_sample=len(chunks))
                        save()
                        if condition == "lora_v2prompt" and item["task"] == "video":
                            baseline_key = sha_text(item["sample_id"])[:20] + f"_base_v2prompt_{window_index:02d}.json"
                            baseline = json.loads((output / "windows" / baseline_key).read_text())
                            memory = baseline["memory"]
                        text = prompt(condition, item["task"], frames, ACTION_CODES, memory)
                        if condition == "lora_v2prompt" and item["task"] == "video" and sha_text(text) != baseline["prompt_sha256"]:
                            raise ValueError("Paired v2 prompts differ; comparison rejected")
                        key = sha_text(item["sample_id"])[:20] + f"_{condition}_{window_index:02d}.json"
                        window_path = output / "windows" / key
                        window_path.parent.mkdir(exist_ok=True)
                        if window_path.exists():
                            window_record = json.loads(window_path.read_text())
                            if window_record["prompt_sha256"] != sha_text(text):
                                raise ValueError("Resume window prompt/memory mismatch")
                        else:
                            images = []
                            for frame in frames:
                                with Image.open(frame["path"]) as image:
                                    images.append(image.convert("RGB"))
                            before = time.monotonic()
                            try:
                                raw = backend.predict(images, text, max_new_tokens=MAX_NEW_TOKENS,
                                                      adapter_enabled=condition == "lora_v2prompt")
                                # Prior frame IDs may support a linked sequence, but
                                # never a future frame not yet supplied to the model.
                                available = [frame["id"] for block in chunks[:window_index + 1] for frame in block]
                                parsed = parse_answer(raw, item["task"], available)
                            except (ValueError, RuntimeError) as error:
                                raw = ""
                                parsed = dict(predicted_label=None, observed_action="unknown",
                                              parse_status="inference_failed", parse_error=str(error))
                            finally:
                                for image in images:
                                    image.close()
                            window_record = dict(sample_id=item["sample_id"], condition=condition,
                                                 window=window_index, raw_response=raw,
                                                 elapsed_sec=round(time.monotonic() - before, 3),
                                                 prompt_sha256=sha_text(text),
                                                 frame_ids=[frame["id"] for frame in frames],
                                                 frame_sha256=[frame["sha256"] for frame in frames],
                                                 memory=memory, **parsed)
                            write_json(window_path, window_record)
                        window_results.append(window_record)
                        # Every entry remains a past model proposal. Keep descriptions
                        # short to stay within the same 1024-token backend budget.
                        if item["task"] == "video":
                            memory = memory + [dict(last_seen_sec=frames[-1]["timestamp"],
                                                    frames=[frame["id"] for frame in frames],
                                                    model_observation=window_record.get("reason", "unknown")[:110],
                                                    current_state="unknown")]
                    if item["task"] == "video":
                        positives = [window for window in window_results if window.get("rule_verdict") == "violation_supported"]
                        parsed = dict(predicted_label=window_results[-1].get("predicted_label"),
                                      rule_verdict="violation_supported" if positives else "insufficient_data",
                                      evidence_frames=list(dict.fromkeys(ref for window in positives for ref in window["evidence_frames"])),
                                      reason=positives[-1]["reason"] if positives else window_results[-1].get("reason", "insufficient_data"),
                                      parse_status="accepted" if all(window["parse_status"] == "accepted" for window in window_results) else "completed_with_rejections")
                    else:
                        parsed = {key: value for key, value in window_results[0].items()
                                  if key in {"observed_action", "description", "review_status", "legacy_two_field_output",
                                             "predicted_label", "parse_status", "parse_error", "wrapper_removed"}}
                    record = dict(sample_id=item["sample_id"], group_id=item["group_id"], task=item["task"],
                                  partition=sample["partition"], condition=condition,
                                  decision_scope="whole_video" if item["task"] == "video" else "supplied_human_interval",
                                  decision_basis="unconfirmed_sparse_video_class_prediction" if item["task"] == "video" else "unconfirmed_action_prediction",
                                  organizer_answers_in_prompt=False,
                                  status="completed" if parsed["parse_status"] != "inference_failed" else "failed",
                                  elapsed_sec=round(time.monotonic() - before_sample, 3),
                                  inference_seconds=sum(window["elapsed_sec"] for window in window_results),
                                  window_records=[sha_text(item["sample_id"])[:20] + f"_{condition}_{index:02d}.json" for index in range(len(chunks))],
                                  prompt_sha256=[window["prompt_sha256"] for window in window_results],
                                  frame_sha256=[frame["sha256"] for frame in item["frames"]], **parsed)
                    with records_path.open("a") as stream:
                        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                    rows.append(record)
                    done.add((item["sample_id"], condition))
                    save()
            status["status"] = "completed"
            save()
    except Exception as error:
        status.update(status="partial" if rows else "failed", error=str(error))
        save()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGTERM, old_term)
        signal.signal(signal.SIGALRM, old_alarm)
        if backend is not None:
            del backend
    print(json.dumps(status, ensure_ascii=False))
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", nargs="+", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--lock", type=Path, default=Path("/home/ubuntu/nuhack-training/.gpu-work.lock"))
    parser.add_argument("--seconds", type=int, default=900)
    parser.add_argument("--task", choices=("video", "action"))
    parser.add_argument("--partition", choices=("inner_holdout", "diagnostic_val"))
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=["base_v2prompt", "lora_v2prompt"])
    parser.add_argument("--resume", action="store_true")
    status = run(parser.parse_args())
    raise SystemExit(0 if status["status"] == "completed" else 2)


if __name__ == "__main__":
    main()
