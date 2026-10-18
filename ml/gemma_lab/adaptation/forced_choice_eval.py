"""Additional whole-video diagnostic: compare digit logits on all16 frozen frames.

This separate protocol does not replace sequential JSON generation or action
evaluation. Its forced binary classes are tentative, NOT evidence-based verdicts.
No label, filename, description, or organizer-specific answer enters the model.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, nullcontext
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import time

from .eval import classify, read_rows
from .hf_backend import MODEL_ID, MODEL_REVISION, MAX_SOFT_TOKENS
from .run_eval import adapter_digest, blind_sample
from .train import DEFAULT_LOCK, assert_gpu_idle, atomic_json, canonical_sha, file_sha, gpu_lock

PROTOCOL = "gemma_all16_digit_logits_diagnostic_v1"
INSTRUCTION = """Inspect all16 silent chronological bank-camera images jointly, including changes
between early and late frames. Make a tentative binary classification of the WHOLE VIDEO:
1 = a possible operational-rule violation; 0 = tentative normal class.
General rules: examine ID before accepting cash; secure the previous customer's cash before
the next customer arrives; put the phone away before counting; distinguish counting from
using an authenticity-checking device; secure the safe, cash, keys, documents and stamp
when the employee leaves; close the room door after entry; secure counted cash before drinking.
Compare the same anonymous people, cash and devices across frames only while continuity
is visible. Unknown objects and roles remain uncertain. Leaving camera view is not necessarily
room departure. A dark screen does not prove shutdown. Recounting is not itself an error.
Missing sampled actions do not prove absence. Do not infer banknote authenticity, identities,
audio, legal findings or unseen actions. This forced choice is only a tentative video classifier,
not proof of a violation or certification that the whole video is normal.
Reply with exactly one digit, 0 or 1, without JSON or explanation."""


def digit_token_ids(tokenizer, rendered_prefix):
    """Fail rather than score a wrong token or silently use multi-token labels."""
    prefix_ids = tokenizer(rendered_prefix, add_special_tokens=False)["input_ids"]
    answer = {}
    for digit in ("0", "1"):
        ids = tokenizer(digit, add_special_tokens=False)["input_ids"]
        if len(ids) != 1 or tokenizer.decode(ids, skip_special_tokens=False) != digit:
            raise ValueError("Digit is not represented by exactly one matching token: " + digit)
        complete = tokenizer(rendered_prefix + digit, add_special_tokens=False)["input_ids"]
        if complete != prefix_ids + ids:
            raise ValueError("Digit tokenization changes at the answer boundary: " + digit)
        answer[digit] = ids[0]
    if answer["0"] == answer["1"]:
        raise ValueError("Candidate digit IDs must differ")
    return answer


def content_for_frames(frames):
    if len(frames) != 16:
        raise ValueError("This diagnostic requires exactly the same16 frozen frames")
    content = [{"type": "text", "text": INSTRUCTION}]
    for index, frame in enumerate(frames):
        # Strict projection: never interpolate group IDs, labels, source text,
        # original filenames, annotation notes or organizer answers.
        content.extend([{"type": "text", "text": f"Frame f{index}, time {float(frame['timestamp']):.3f} seconds:"},
                        {"type": "image"}])
    return content


def encode_all_frames(backend, blind):
    from PIL import Image
    content = content_for_frames(blind["frames"])
    text = backend.processor.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    )
    candidates = digit_token_ids(backend.processor.tokenizer, text)
    images = []
    for frame in blind["frames"]:
        with Image.open(frame["path"]) as source:
            images.append(source.convert("RGB"))
    inputs = dict(backend.processor(
        text=[text], images=[images], return_tensors="pt", padding=False,
        images_kwargs={"max_soft_tokens": MAX_SOFT_TOKENS},
        text_kwargs={"add_special_tokens": False},
    ))
    tokens = int(inputs["input_ids"].shape[1])
    if tokens > 2048:
        raise ValueError(f"All16-frame input exceeds frozen2048-token cap: {tokens}; no truncation applied")
    return inputs, candidates, tokens, hashlib.sha256(text.encode()).hexdigest()


def label_from_logits(scores):
    if set(scores) != {"0", "1"} or not all(isinstance(v, (float, int)) and math.isfinite(v) for v in scores.values()):
        raise ValueError("Both candidate logits must be finite")
    return 1 if scores["1"] > scores["0"] else 0


def score(backend, inputs, candidate_ids, adapter_enabled):
    import torch
    context = nullcontext() if adapter_enabled else backend.model.disable_adapter()
    backend.model.eval()
    with context, torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        output = backend.model(**inputs, use_cache=False, logits_to_keep=1)
    logits = output.logits[0, -1, :].float()
    result = {digit: float(logits[token_id].item()) for digit, token_id in candidate_ids.items()}
    return label_from_logits(result), result


def metrics(samples, predictions):
    targets = [dict(group_id=row["group_id"], video=row["video"], split=row["split"],
                    label=row["reference"]["label"], label_scope="whole_video") for row in samples]
    return {
        "protocol": PROTOCOL, "additional_protocol_not_replacement": True,
        "classification_scope": "tentative_whole_video_binary_class_not_rule_evidence",
        "quality_of_actions_or_temporal_boundaries_measured": False,
        "independent_test": False, "threshold_tuned": False,
        "conditions": {condition: classify(targets, [p for p in predictions if p["condition"] == condition])
                       for condition in ("base_same_prompt", "lora_same_prompt")},
    }


def write_predictions(path, predictions):
    temporary = path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in predictions:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def run(args):
    started = time.monotonic()
    root, output = args.run_root.resolve(), args.output.resolve()
    if not output.is_relative_to(root):
        raise ValueError("Diagnostic output must remain in the isolated adaptation run")
    if output.exists():
        raise FileExistsError("Preserve completed diagnostics; choose a new output directory")
    if not 60 <= args.seconds <= 300:
        raise ValueError("This diagnostic is bounded to60–300 seconds")
    protocol_path = root / "frozen_group_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    if protocol["protocol_sha256"] != canonical_sha({k: v for k, v in protocol.items() if k != "protocol_sha256"}):
        raise ValueError("Frozen protocol checksum mismatch")
    rows = read_rows(args.samples)
    samples = [r for r in rows if r["partition"] == args.partition and r["task"] == "video"]
    expected = set(protocol["inner_holdout_groups"] if args.partition == "inner_holdout" else protocol["diagnostic_val_groups"])
    if len(samples) != len(expected) or {r["group_id"] for r in samples} != expected or len(expected) != 8:
        raise ValueError("Schedule must include all8 frozen groups exactly once")
    for row in samples:
        identity = protocol["source_identity"][row["group_id"]]
        if row["video_sha256"] != identity["video_sha256"] or row["split"] != identity["split"]:
            raise ValueError("Source group identity mismatch")
        if row.get("organizer_answers_in_prompt") or row.get("labels_in_prompt"):
            raise ValueError("Evaluation sample marked as contaminated")
    blind = {row["sample_id"]: blind_sample(row, root) for row in samples}
    output.mkdir(parents=True)
    config = {
        "protocol": PROTOCOL, "additional_protocol_not_replacement": True,
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "adapter_checksums": adapter_digest(args.adapter),
        "samples_sha256": file_sha(args.samples), "frozen_protocol_file_sha256": file_sha(protocol_path),
        "original_manifest_sha256": protocol["original_manifest_sha256"],
        "instruction_sha256": hashlib.sha256(INSTRUCTION.encode()).hexdigest(),
        "evaluator_code_sha256": file_sha(Path(__file__)),
        "backend_code_sha256": file_sha(Path(__file__).with_name("hf_backend.py")),
        "sample_ids": [r["sample_id"] for r in samples], "partition": args.partition,
        "frame_sha256": {k: [f["sha256"] for f in v["frames"]] for k, v in blind.items()},
        "input_protocol": "same16 frozen native frames jointly;70soft tokens/image;2048token cap",
        "classification": "higher next-token logit among exact single-token0/1; tie chooses0",
        "scores_are_calibrated_probabilities": False, "threshold_tuned": False,
        "organizer_answers_in_prompt": False, "sequential_json_and_action_results_preserved": True,
    }
    atomic_json(output / "config.json", config)
    predictions, backend, device_inputs = [], None, None
    status = {"status": "processing", "phase": "waiting_gpu", "scheduled_groups": 8, "completed_pairs": 0}
    guard = ExitStack()

    def save():
        status["elapsed_sec"] = time.monotonic() - started
        status["completed_pairs"] = len(predictions) // 2
        atomic_json(output / "status.json", status)
        write_predictions(output / "predictions.jsonl", predictions)
        atomic_json(output / "metrics.json", metrics(samples, predictions))

    def interrupted(signum, frame):
        raise TimeoutError("Forced-choice diagnostic deadline/interruption; completed pairs preserved")

    prior = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGALRM, signal.SIGTERM)}
    signal.setitimer(signal.ITIMER_REAL, max(0.01, args.seconds - (time.monotonic() - started)))
    try:
        save()
        for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
            os.environ[key] = "1"
        os.environ["HF_HOME"] = str(args.model.resolve().parent / "training_hf_cache")
        guard.enter_context(gpu_lock(args.gpu_lock))
        assert_gpu_idle()
        from .hf_backend import load_backend
        import torch
        status["phase"] = "loading_model"
        save()
        backend = load_backend(args.model, training=False, adapter_path=args.adapter)
        status["backend"] = backend.provenance()
        for sample in samples:
            if time.monotonic() - started >= args.seconds - 10:
                raise TimeoutError("Reserved cleanup grace reached")
            item = blind[sample["sample_id"]]
            status.update(phase="scoring_pair", sample_id=sample["sample_id"])
            save()
            pair = []
            try:
                inputs, candidates, tokens, prompt_hash = encode_all_frames(backend, item)
                device_inputs = {k: v.to(backend.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
                for condition, enabled in (("base_same_prompt", False), ("lora_same_prompt", True)):
                    tick = time.monotonic()
                    try:
                        label, scores = score(backend, device_inputs, candidates, enabled)
                        error = None
                    except (ValueError, RuntimeError) as exc:
                        label, scores, error = None, None, type(exc).__name__ + ": " + str(exc)
                    pair.append(dict(sample_id=sample["sample_id"], group_id=sample["group_id"],
                                     condition=condition, predicted_label=label, decision_scope="whole_video",
                                     decision_basis="forced_choice_digit_logit_tentative_classifier",
                                     status="completed" if error is None else "failed", error=error,
                                     candidate_token_ids=candidates, candidate_logits=scores,
                                     scores_are_calibrated_probabilities=False, input_tokens=tokens,
                                     frame_sha256=[f["sha256"] for f in item["frames"]], prompt_sha256=prompt_hash,
                                     organizer_answers_in_prompt=False, elapsed_sec=time.monotonic() - tick,
                                     peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                     rule_verdict="insufficient_data", review_status="proposed"))
            except ValueError as exc:
                for condition in ("base_same_prompt", "lora_same_prompt"):
                    pair.append(dict(sample_id=sample["sample_id"], group_id=sample["group_id"], condition=condition,
                                     predicted_label=None, decision_scope="whole_video", status="failed",
                                     error=str(exc), organizer_answers_in_prompt=False,
                                     frame_sha256=[f["sha256"] for f in item["frames"]]))
            predictions.extend(pair)
            device_inputs = None
            save()
        status.update(status="completed", phase="finished")
    except TimeoutError as exc:
        status.update(status="partial", phase="finished", error=str(exc))
    except BaseException as exc:
        status.update(status="failed", phase="failed", error=type(exc).__name__ + ": " + str(exc))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        try:
            save()
        finally:
            try:
                device_inputs = None
                if backend is not None:
                    backend.model = None
                    backend = None
                    gc.collect()
                    import torch
                    torch.cuda.empty_cache()
            finally:
                guard.close()
                for sig, handler in prior.items():
                    signal.signal(sig, handler)
    print(json.dumps(status, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--partition", choices=("inner_holdout", "diagnostic_val"), default="inner_holdout")
    parser.add_argument("--seconds", type=int, default=300)
    parser.add_argument("--gpu-lock", type=Path, default=DEFAULT_LOCK)
    run(parser.parse_args())
