"""Recompute evaluation readiness from saved predictions without model inference.

Run: python ml/readiness_audit.py --output runs/readiness/report.json

Optional event evaluation accepts a JSON truth object with a ``videos`` list:
  {"video": "a.mp4", "human_confirmed": true, "coverage_complete": true,
   "events": [{"start_sec": 1, "end_sec": 3, "type": "cash_left",
               "human_confirmed": true}]}
Predictions use the same video/event structure without confirmation flags.
Only exhaustively human-reviewed videos may contribute false-positive counts.
Proposals, approximate source anchors and absent annotations are not ground truth.
"""

import argparse
import csv
import hashlib
import json
import math
import signal
import time
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def wilson_interval(successes, count, z=1.959963984540054):
    """Binomial interval; callers must state when independence is unverified."""
    if count < 0 or successes < 0 or successes > count:
        raise ValueError("Invalid binomial counts")
    if not count:
        return None
    p = successes / count
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    margin = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def binary_metrics(targets, decisions):
    if len(targets) != len(decisions):
        raise ValueError("Mismatched targets and predictions")
    if any(v not in (0, 1) for v in list(targets) + list(decisions)):
        raise ValueError("Only binary targets and decisions are valid")
    counts = dict(tp=0, tn=0, fp=0, fn=0)
    for target, decision in zip(targets, decisions):
        counts[{(1, 1): "tp", (0, 0): "tn", (0, 1): "fp", (1, 0): "fn"}[(target, decision)]] += 1
    tp, tn, fp, fn = (counts[k] for k in ("tp", "tn", "fp", "fn"))
    recall, specificity = ratio(tp, tp + fn), ratio(tn, tn + fp)
    return dict(
        count=len(targets), confusion=counts,
        accuracy=ratio(tp + tn, len(targets)), precision=ratio(tp, tp + fp),
        recall=recall, specificity=specificity,
        f1=ratio(2 * tp, 2 * tp + fp + fn),
        balanced_accuracy=(recall + specificity) / 2 if recall is not None and specificity is not None else None,
        false_positive_rate=ratio(fp, tn + fp),
    )


def validate_predictions(rows, expected, threshold, *, windows=False):
    """Match every prediction to frozen source labels; reject silent exclusions."""
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Invalid decision threshold")
    found, targets, decisions = set(), [], []
    for row in rows:
        key = (row["video"], float(row["start_sec"])) if windows else row["video"]
        if key in found:
            raise ValueError("Duplicate prediction evaluation unit")
        found.add(key)
        if key not in expected or row["target"] != expected[key]:
            raise ValueError("Prediction target differs from frozen manifest")
        score = float(row.get("probability", row.get("score")))
        if not math.isfinite(score):
            raise ValueError("Non-finite prediction score")
        targets.append(row["target"])
        decisions.append(int(score >= threshold))
    if found != set(expected):
        raise ValueError("Prediction coverage differs from complete evaluation set")
    return binary_metrics(targets, decisions)


def _check_report_metrics(recomputed, declared):
    for key in ("accuracy", "precision", "recall", "f1", "balanced_accuracy", "confusion"):
        if key not in declared:
            continue
        if key == "confusion":
            if recomputed[key] != declared[key]:
                raise ValueError("Saved confusion differs from predictions")
        elif recomputed[key] is not None and not math.isclose(recomputed[key], declared[key], abs_tol=1e-7):
            raise ValueError(f"Saved {key} differs from predictions")


def validate_full_video_windows(prediction, expected_spans):
    """Ensure a saved max score covered every frozen whole-video window."""
    scores = prediction.get("windows", [])
    if len(scores) != len(expected_spans) or prediction.get("window_count") != len(scores) or not scores:
        raise ValueError("Incomplete whole-video window coverage")
    for score, span in zip(scores, expected_spans):
        if not all(math.isclose(float(score[key]), float(span[key]), abs_tol=1e-6) for key in ("start_sec", "end_sec")):
            raise ValueError("Whole-video window boundaries differ from the frozen plan")
        if not math.isfinite(float(score["score"])):
            raise ValueError("Non-finite window score")
    if max(s["score"] for s in scores) != prediction["score"]:
        raise ValueError("Saved max aggregation differs from its windows")


def _interval(event):
    start, end = float(event["start_sec"]), float(event["end_sec"])
    if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
        raise ValueError("Event interval must be finite, positive-length and nonnegative")
    return start, end


def temporal_iou(a, b):
    a0, a1 = _interval(a)
    b0, b1 = _interval(b)
    overlap = max(0.0, min(a1, b1) - max(a0, b0))
    return overlap / (a1 - a0 + b1 - b0 - overlap)


def _match_events(truth, predictions, threshold, typed=False):
    """Maximum-cardinality bipartite matching, deterministic IoU-first traversal.

    This avoids counting two detections of one event as two true positives.
    It is not an optimal-boundary-error assignment; timing metrics state that.
    """
    edges = []
    for prediction in predictions:
        candidates = []
        for i, target in enumerate(truth):
            overlap = temporal_iou(target, prediction)
            if overlap >= threshold and (not typed or target["type"] == prediction.get("type")):
                candidates.append((overlap, i))
        edges.append([i for _, i in sorted(candidates, key=lambda item: (-item[0], item[1]))])
    assigned = {}

    def augment(prediction_index, visited):
        for truth_index in edges[prediction_index]:
            if truth_index in visited:
                continue
            visited.add(truth_index)
            if truth_index not in assigned or augment(assigned[truth_index], visited):
                assigned[truth_index] = prediction_index
                return True
        return False

    for prediction_index in range(len(predictions)):
        augment(prediction_index, set())
    return sorted(assigned.items())


def evaluate_events(truth, predictions, iou_threshold=0.5):
    """Offline metrics gated on complete human annotations, not model proposals."""
    if not 0 < iou_threshold <= 1:
        raise ValueError("Temporal IoU threshold must be in (0, 1]")
    unavailable = dict(status="unavailable", metrics=None)
    truth_videos = truth.get("videos", [])
    if not truth_videos:
        return dict(unavailable, reason="No human-confirmed event ground truth supplied")
    by_video = {}
    for video in truth_videos:
        if video["video"] in by_video:
            raise ValueError("Duplicate ground-truth video")
        if video.get("human_confirmed") is not True or video.get("coverage_complete") is not True:
            return dict(unavailable, reason="Every evaluated video needs complete human-confirmed event coverage")
        for event in video.get("events", []):
            _interval(event)
            if event.get("human_confirmed") is not True:
                return dict(unavailable, reason="Unconfirmed proposals cannot be event ground truth")
        by_video[video["video"]] = video
    prediction_by_video = {}
    for video in predictions.get("videos", []):
        if video["video"] not in by_video or video["video"] in prediction_by_video:
            raise ValueError("Unknown or duplicate prediction video")
        for event in video.get("events", []):
            _interval(event)
        prediction_by_video[video["video"]] = video
    if set(prediction_by_video) != set(by_video):
        raise ValueError("An explicit prediction result is required for every reviewed video")

    true_positives = false_positives = false_negatives = 0
    boundary_errors = []
    classes = defaultdict(lambda: dict(tp=0, fp=0, fn=0))
    typed_ground_truth = all(
        isinstance(e.get("type"), str) and bool(e["type"].strip()) and e["type"] != "violation_unspecified"
        for v in truth_videos for e in v.get("events", [])
    )
    targets, decisions = [], []
    for video_id, video in by_video.items():
        expected, observed = video.get("events", []), prediction_by_video[video_id].get("events", [])
        targets.append(int(bool(expected)))
        decisions.append(int(bool(observed)))
        matches = _match_events(expected, observed, iou_threshold)
        true_positives += len(matches)
        false_positives += len(observed) - len(matches)
        false_negatives += len(expected) - len(matches)
        for a, b in matches:
            t, p = _interval(expected[a]), _interval(observed[b])
            boundary_errors.append((abs(t[0] - p[0]) + abs(t[1] - p[1])) / 2)
        if typed_ground_truth:
            typed_matches = _match_events(expected, observed, iou_threshold, typed=True)
            matched_true = {a for a, _ in typed_matches}
            matched_pred = {b for _, b in typed_matches}
            for i, event in enumerate(expected):
                classes[event["type"]]["tp" if i in matched_true else "fn"] += 1
            for i, event in enumerate(observed):
                if i not in matched_pred:
                    classes[event.get("type") or "__untyped_prediction__"]["fp"] += 1
    tp, fp, fn = true_positives, false_positives, false_negatives
    class_results = {
        name: dict(**c, f1=ratio(2 * c["tp"], 2 * c["tp"] + c["fp"] + c["fn"]))
        for name, c in sorted(classes.items())
    } if typed_ground_truth else None
    macro = (sum(c["f1"] for c in class_results.values()) / len(class_results)) if class_results else None
    return dict(
        status="evaluated", temporal_iou_threshold=iou_threshold,
        matching="maximum_cardinality_one_to_one; deterministic IoU-first traversal",
        metrics=dict(
            video_verdict=binary_metrics(targets, decisions),
            event_confusion=dict(tp=tp, fp=fp, fn=fn),
            event_precision=ratio(tp, tp + fp), event_recall=ratio(tp, tp + fn),
            event_f1=ratio(2 * tp, 2 * tp + fp + fn),
            false_positives_per_video=fp / len(by_video),
            matched_boundary_mean_absolute_error_sec=ratio(sum(boundary_errors), len(boundary_errors)),
            type_macro_f1=macro, types=class_results,
        ),
        limitations=[
            "Boundary error is conditional on matched events; misses are measured by event recall.",
            "Type macro-F1 uses one-to-one detections of the same type and IoU, over observed label union.",
            "Annotation confirmation is a caller provenance assertion, not an automatic semantic audit.",
        ],
    )


def evaluate_x3d_videos(root=ROOT, deadline_seconds=300):
    """Bounded CPU rerun of the saved API baseline on the reused video holdout."""
    root = Path(root)
    started = time.monotonic()
    output = root / "runs/readiness/x3d_video_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    status_path = output.with_name("x3d_video_status.json")

    def write(path, value):
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)

    def check_deadline(*_):
        if time.monotonic() - started >= deadline_seconds:
            raise TimeoutError("CPU audit exceeded its whole-run deadline")

    previous_handler = signal.signal(signal.SIGALRM, check_deadline)
    signal.setitimer(signal.ITIMER_REAL, deadline_seconds)
    predictions = []
    try:
        # Import only in explicit evaluation mode; the report generator is stdlib-only.
        import av
        import torch
        from infer_video import windows
        from model_suite import MODELS, load_bundle, load_frames, predict

        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        manifest_path = root / "data/training/manifest.csv"
        manifest_hash = digest(manifest_path)
        with manifest_path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        by_sha = {}
        for row in rows:
            if row["split"] != "val":
                continue
            if row["sha256"] in by_sha:
                if by_sha[row["sha256"]]["target"] != row["target"]:
                    raise ValueError("Conflicting validation source labels")
            else:
                by_sha[row["sha256"]] = row
        if len(by_sha) != 8:
            raise ValueError("This frozen audit expects exactly eight SHA-unique validation videos")
        model_path = root / "artifacts/models/x3d/model.pt"
        model_hash = digest(model_path)
        load_start = time.monotonic()
        extractor, bundle = load_bundle(model_path, root / "artifacts/assets", "cpu")
        load_seconds = time.monotonic() - load_start
        if bundle["manifest_sha256"] != manifest_hash or bundle["model_key"] != "x3d":
            raise ValueError("Checkpoint does not match the frozen dataset and model")
        threshold = float(bundle["threshold"])
        cfg = MODELS["x3d"]
        total_windows = 0
        for row in by_sha.values():
            check_deadline()
            path = root / "data/raw/open" / row["video"]
            if digest(path) != row["sha256"]:
                raise ValueError("Validation video bytes differ from the frozen manifest")
            with av.open(str(path)) as container:
                track = container.streams.video[0]
                duration = float(track.duration * track.time_base) if track.duration is not None else float(container.duration / av.time_base)
            spans = windows(duration, length=8, stride=4)
            scores = []
            video_started = time.monotonic()
            for start, end in spans:
                check_deadline()
                window_started = time.monotonic()
                frames = load_frames(path, start, end, cfg["frames"], cfg["size"])
                score = predict(extractor, bundle, frames.unsqueeze(0))[0]
                check_deadline()
                if not math.isfinite(score):
                    raise ValueError("Non-finite model score")
                scores.append(dict(start_sec=start, end_sec=end, score=score, elapsed_sec=time.monotonic() - window_started))
                total_windows += 1
                write(status_path, dict(status="running", completed_videos=len(predictions), total_videos=8, completed_windows=total_windows, elapsed_sec=time.monotonic() - started))
                write(output.with_name("x3d_video_partial.json"), dict(
                    status="partial", model_sha256=model_hash, manifest_sha256=manifest_hash,
                    completed_predictions=predictions,
                    current_video=dict(video=row["video"], target=int(row["target"]), sha256=row["sha256"], windows=scores),
                    elapsed_sec=time.monotonic() - started,
                ))
            predictions.append(dict(
                video=row["video"], group_id=row["group_id"], sha256=row["sha256"],
                target=int(row["target"]), score=max(s["score"] for s in scores),
                duration_sec=duration, windows=scores, window_count=len(scores),
                elapsed_sec=time.monotonic() - video_started,
            ))
            print(json.dumps(dict(model="x3d", completed_videos=len(predictions), total_videos=8, completed_windows=total_windows, elapsed_sec=time.monotonic() - started)), flush=True)
        check_deadline()
        if digest(model_path) != model_hash or digest(manifest_path) != manifest_hash:
            raise ValueError("Model or manifest changed during CPU evaluation")
        result = dict(
            schema_version=1, status="completed", model="x3d", scope="whole_video",
            device="cpu", torch_threads=2, torch_interop_threads=1,
            model_sha256=model_hash, manifest_sha256=manifest_hash,
            threshold=threshold, threshold_selection="unchanged saved checkpoint threshold; no tuning in this evaluation",
            aggregation="max", window_sec=8, stride_sec=4,
            preprocessing=dict(frames=cfg["frames"], size=cfg["size"], resize="letterbox"),
            checked_videos=len(predictions), checked_windows=total_windows,
            load_seconds=load_seconds, total_seconds=time.monotonic() - started,
            metrics=validate_predictions(predictions, {row["video"]: int(row["target"]) for row in by_sha.values()}, threshold),
            predictions=predictions,
            source_sha_verified=True, independent_test=False,
            code_sha256={name: digest(root / name) for name in ["ml/readiness_audit.py", "ml/model_suite.py", "ml/infer_video.py"]},
            limitations=[
                "Eight repeatedly used validation videos; no independent generalization estimate.",
                "Full-video binary verdict only; event times, types and legal rules are not validated.",
                "Timing covers saved-model inference and report bookkeeping, not HTTP upload, queueing or evidence encoding.",
                "Only this checkpoint and preprocessing were measured; no threshold or model selection performed here.",
            ],
        )
        write(output, result)
        write(status_path, dict(status="completed", completed_videos=8, completed_windows=total_windows, elapsed_sec=time.monotonic() - started))
        return result
    except Exception as exc:
        write(status_path, dict(status="failed", completed_videos=len(predictions), elapsed_sec=time.monotonic() - started, error=str(exc)))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def build_audit(root=ROOT):
    root = Path(root)
    inputs = {}

    def read(relative):
        path = root / relative
        inputs[relative] = digest(path)
        return load_json(path)

    def read_csv(relative):
        path = root / relative
        inputs[relative] = digest(path)
        with path.open(encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))

    manifest_path = "data/training/manifest.csv"
    manifest = read_csv(manifest_path)
    source_rows = read_csv("data/review/event_review.csv")
    summary = read("data/training/summary.json")
    release = read("data/autolabel/release.json")
    validation = read("data/autolabel/validation_report.json")
    contract = read("data/autolabel/TRAINING_CONTRACT.json")
    plan = read("runs/quality_v2/plan.json")
    manifest_hash = inputs[manifest_path]
    if summary["manifest_sha256"] != manifest_hash or plan["manifest_sha256"] != manifest_hash:
        raise ValueError("Evaluation plan or summary refers to a different manifest")
    protected = {}
    for relative, expected_hash in release["protected_inputs"].items():
        current = digest(root / relative)
        inputs[relative] = current
        protected[relative] = current == expected_hash
    if not all(protected.values()):
        raise ValueError("Frozen supervision inputs changed after dataset release")
    groups, videos = {}, {}
    split_by_sha = defaultdict(set)
    for row in manifest:
        split_by_sha[row["sha256"]].add(row["split"])
        signature = row["split"], int(row["target"]), row["sha256"]
        if row["group_id"] in groups and groups[row["group_id"]] != signature:
            raise ValueError("Conflicting source group labels or splits")
        if row["video"] in videos and videos[row["video"]] != signature:
            raise ValueError("Conflicting source video labels or splits")
        groups[row["group_id"]] = signature
        videos[row["video"]] = signature
    if any(len(splits) > 1 for splits in split_by_sha.values()):
        raise ValueError("Exact source SHA occurs in multiple splits")
    if len(groups) != summary["unique_videos"]:
        raise ValueError("Summary source count differs from manifest")
    expected_videos = {v: target for v, (split, target, _) in videos.items() if split == "val"}
    expected_windows = {(r["video"], float(r["start_sec"])): int(r["target"]) for r in manifest if r["split"] == "val"}
    if len(expected_windows) != sum(r["split"] == "val" for r in manifest):
        raise ValueError("Manifest contains duplicate evaluation windows")

    evaluations = []
    files = [
        ("videomaev2", "runs/quality_v2/baseline_video_report.json"),
        ("whole_video_mil", "runs/quality_v2/report.json"),
        ("whole_video_retrieval", "runs/quality_v2/retrieval_report.json"),
        ("whole_video_sequence", "runs/quality_v2/sequence_report.json"),
        ("whole_video_semantic", "runs/quality_v2/semantic_report.json"),
    ]
    for model, relative in files:
        report = read(relative)
        if report.get("manifest_sha256") != manifest_hash:
            raise ValueError(f"Stale evaluation report: {relative}")
        is_baseline = model == "videomaev2"
        threshold = report["threshold"] if is_baseline else report["selected"]["threshold"]
        predictions = report["predictions"] if is_baseline else report["holdout_videos"]
        metrics = validate_predictions(predictions, expected_videos, threshold)
        _check_report_metrics(metrics, report["metrics"] if is_baseline else report["holdout_video_metrics"])
        c = metrics["confusion"]
        evaluations.append(dict(
            model=model, scope="whole_video", source=relative, threshold=threshold,
            metrics=metrics,
            accuracy_wilson_95_descriptive=wilson_interval(c["tp"] + c["tn"], metrics["count"]),
            interval_is_generalization_claim=False,
            all_videos_predicted_positive=(c["tn"] + c["fn"] == 0),
            independent_test=False,
        ))

    x3d_path = "runs/readiness/x3d_video_report.json"
    if (root / x3d_path).is_file():
        x3d = read(x3d_path)
        if x3d.get("status") != "completed" or x3d.get("checked_videos") != len(expected_videos):
            raise ValueError("X3D full-video audit is incomplete")
        if x3d.get("manifest_sha256") != manifest_hash or x3d.get("model_sha256") != digest(root / "artifacts/models/x3d/model.pt"):
            raise ValueError("X3D audit refers to a different manifest or checkpoint")
        if x3d.get("source_sha_verified") is not True:
            raise ValueError("X3D audit did not verify raw validation source SHA")
        bags_by_video = {b["video"]: b for b in plan["bags"] if b["split"] == "val"}
        for prediction in x3d["predictions"]:
            if prediction.get("sha256") != videos[prediction["video"]][2]:
                raise ValueError("X3D prediction SHA differs from the manifest")
            expected_spans = [plan["clips"][i] for i in bags_by_video[prediction["video"]]["indices"]]
            validate_full_video_windows(prediction, expected_spans)
        if x3d.get("checked_windows") != sum(p["window_count"] for p in x3d["predictions"]):
            raise ValueError("X3D checked-window count differs from predictions")
        metrics = validate_predictions(x3d["predictions"], expected_videos, x3d["threshold"])
        _check_report_metrics(metrics, x3d["metrics"])
        c = metrics["confusion"]
        evaluations.append(dict(
            model="x3d", scope="whole_video", source=x3d_path, threshold=x3d["threshold"],
            metrics=metrics, model_sha256=x3d["model_sha256"], device="cpu",
            accuracy_wilson_95_descriptive=wilson_interval(c["tp"] + c["tn"], metrics["count"]),
            interval_is_generalization_claim=False,
            all_videos_predicted_positive=(c["tn"] + c["fn"] == 0),
            independent_test=False,
        ))

    legacy_path = "artifacts/models/videomaev2/validation.json"
    legacy = read(legacy_path)
    settings = read("artifacts/models/videomaev2/settings.json")
    window_metrics = validate_predictions(legacy["predictions"], expected_windows, settings["threshold"], windows=True)
    _check_report_metrics(window_metrics, legacy["metrics"])

    pairs_path = contract["weak_video_text"]["artifact"]
    inputs[pairs_path] = digest(root / pairs_path)
    pairs = [json.loads(line) for line in (root / pairs_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    source_groups = {row["video"]: row["group_id"] for row in source_rows}
    for pair in pairs:
        if pair["group_id"] not in groups or groups[pair["group_id"]][0] != "train" or pair["split"] != "train":
            raise ValueError("Weak training pair crosses frozen train split")
        # Narration may refer to the second filename of an exact-duplicate group.
        # The immutable source CSV preserves those aliases; manifest picks one.
        if source_groups.get(pair["video"]) != pair["group_id"]:
            raise ValueError("Weak training pair refers to an unknown/non-train source")
    if len(pairs) != release["user_video_text_pairs"] or len(pairs) != validation["user_pairs"]:
        raise ValueError("Weak pair count differs between release and actual artifact")
    strong_event_count = release["positive_interval_ground_truth_created"]
    box_count = release["human_verified_boxes"]
    if strong_event_count != validation["positive_interval_ground_truth_created"] or box_count != validation["human_object_boxes_created"]:
        raise ValueError("Dataset reports disagree on confirmed annotation counts")

    split_counts = {}
    for split in sorted({row["split"] for row in manifest}):
        split_counts[split] = dict(
            sha_unique_videos=sum(s == split for s, _, _ in groups.values()),
            windows=sum(row["split"] == split for row in manifest),
            window_targets=dict(Counter(str(row["target"]) for row in manifest if row["split"] == split)),
            video_targets=dict(Counter(str(target) for s, target, _ in groups.values() if s == split)),
        )
    claims = [
        dict(claim="The five pre-review whole-video reports show 37.5–50% accuracy", status="confirmed", qualification="Descriptive result on a reused holdout, not a population accuracy estimate. The separately rerun X3D baseline, if available, is listed independently."),
        dict(claim="Base VideoMAE and MIL predict every evaluated video positive", status="confirmed", qualification="Observed on these eight videos at their saved thresholds; not proof every possible video receives a positive verdict."),
        dict(claim="Forty independent videos remain after exact deduplication", status="needs_qualification", qualification="Forty SHA-unique sources; independent scene/session/actor groups have not been established."),
        dict(claim="Autolabel added no supervision", status="overstated", qualification="No new human-confirmed strong intervals or boxes, but 99 train-only weak human video-text pairs exist alongside organizer video labels."),
        dict(claim="DINO cosine proves cross-split scene leakage", status="not_established", qualification="Embedding similarity is a review signal, not scene/session identity. The current SHA grouping leaves related-scene leakage risk unresolved."),
        dict(claim="All recent experiment thresholds were tuned on the holdout", status="needs_qualification", qualification="quality_v2 thresholds use train-only nested selection; the same holdout was still viewed across methods and is not an independent final test."),
        dict(claim="Training with the current supervision is impossible", status="overstated", qualification="Weak/video-level training is possible; precise temporal/object/type claims cannot be validated from unconfirmed proposals."),
        dict(claim="An engineering score of 13/100 is established", status="not_measured", qualification="No rubric-level adjudication or independent competition evaluation was performed by this audit."),
    ]
    return dict(
        schema_version=1, audit_kind="saved_prediction_and_supervision_readiness",
        target=dict(metric="user_requested_quality_target_not_a_measured_result", range=[0.85, 0.90], achieved=False),
        dataset=dict(
            source_files=len({row["video"] for row in source_rows}), sha_unique_videos=len(groups),
            removed_exact_duplicate_files=len({row["video"] for row in source_rows}) - len(groups),
            windows=len(manifest), splits=split_counts,
            independent_scene_count=None, scene_grouping_verified=False,
            exact_sha_split_isolation=True,
            reused_holdout_videos=len(expected_videos), independent_test_videos=0,
            accuracy_resolution=1 / len(expected_videos) if expected_videos else None,
        ),
        whole_video_evaluations=evaluations,
        organizer_window_evaluation=dict(
            model="videomaev2", scope="organizer_windows", source=legacy_path,
            threshold=settings["threshold"], metrics=window_metrics,
            target_definition="Organizer positive anchor contexts; source-normal windows. Not exact event intervals.",
            comparable_to_whole_video_metrics=False,
        ),
        supervision=dict(
            organizer_video_labels=len(groups), weak_human_video_text_pairs=len(pairs),
            weak_pairs_eligible=sum(p.get("eligible_for_weak_video_text_training") is True for p in pairs),
            weak_pairs_strong_event_eligible=sum(p.get("eligible_for_strong_event_training") is True for p in pairs),
            human_confirmed_positive_intervals_created=strong_event_count,
            human_confirmed_object_boxes_created=box_count,
            confirmed_counts_scope="Dataset release and its validation report; not an exhaustive scan of future user imports.",
            legacy_review_status_counts=dict(Counter(row["review_status"] for row in source_rows)),
            model_proposals_are_ground_truth=False,
        ),
        event_evaluation=dict(status="unavailable", metrics=None, reason="No complete human-confirmed event evaluation set supplied; proposals and organizer anchors are not event ground truth."),
        review_claims=claims,
        statistical_limitations=[
            "Wilson intervals describe binomial uncertainty only under independent Bernoulli trials; scene independence is unverified.",
            "Repeated holdout inspection and model selection are not corrected by a Wilson interval.",
            "Correlated overlapping windows are not additional independent videos.",
            "High window false-positive rate plus max pooling is concerning; an independence-based accumulation formula is not justified for overlapping windows.",
            "A fixed sample count such as 200–300 does not itself guarantee unbiased quality; scene diversity, test isolation and target metric matter.",
        ],
        illustrative_7_of_8_accuracy=dict(value=0.875, wilson_95_descriptive=wilson_interval(7, 8), hypothetical=True, independent_test_claim=False),
        protected_inputs_unchanged=protected, input_sha256=inputs,
        raw_video_files_rehashed=False,
        scope_exclusions=["API conformance", "license approval", "legal rule correctness", "raw video inference rerun", "semantic human review"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=Path("runs/readiness/report.json"))
    parser.add_argument("--event-ground-truth", type=Path)
    parser.add_argument("--event-predictions", type=Path)
    parser.add_argument("--evaluate-x3d", action="store_true", help="Explicit bounded CPU whole-video rerun; never runs during report-only mode")
    args = parser.parse_args()
    if bool(args.event_ground_truth) != bool(args.event_predictions):
        parser.error("Event truth and predictions must be provided together")
    if args.evaluate_x3d:
        evaluate_x3d_videos(args.root)
    report = build_audit(args.root)
    if args.event_ground_truth:
        report["event_evaluation"] = evaluate_events(load_json(args.event_ground_truth), load_json(args.event_predictions))
        for path in (args.event_ground_truth, args.event_predictions):
            report["input_sha256"][str(path)] = digest(path)
    output = args.output if args.output.is_absolute() else args.root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(dict(output=str(output), video_accuracy={r["model"]: r["metrics"]["accuracy"] for r in report["whole_video_evaluations"]}, event_metrics=report["event_evaluation"]["status"])))


if __name__ == "__main__":
    main()
