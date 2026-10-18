"""Offline evaluation with grouped splits and explicit model abstention.

No model calls, no threshold fitting, and no mutation of source labels. Existing
val is a reused diagnostic set, never described as an independent test. Sparse
human events support reference recovery; they do not establish false positives.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

VERSION = "gemma_adaptation_evaluation_v1"
LABEL_SOURCE = "organizer_original_video_label"
ABSTAIN = {"insufficient_data", "not_applicable", "unknown", "unassessed"}


def read_rows(path: Path) -> list[dict]:
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as stream:
            return list(csv.DictReader(stream))
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    value = json.loads(path.read_text())
    if not isinstance(value, list):
        raise ValueError("Expected a JSON list or JSONL records")
    return value


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def binary(value: Any) -> int:
    if value in (0, "0", "normal"):
        return 0
    if value in (1, "1", "violation", "violation_unspecified"):
        return 1
    raise ValueError(f"Unknown video label: {value!r}")


def video_targets(rows: list[dict]) -> list[dict]:
    """Collapse manifest windows/aliases to one ORIGINAL whole-video label/group.

    This does not create window labels. Reject inconsistent labels or splits.
    """
    groups: dict[str, dict] = {}
    video_groups: dict[str, str] = {}
    hash_groups: dict[str, str] = {}
    for item in rows:
        group = str(item["group_id"])
        if not group:
            raise ValueError("Missing source group_id")
        split = item["split"]
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown fixed split: {split}")
        if item.get("label_scope", "whole_video") != "whole_video":
            raise ValueError("Whole-video metrics require whole-video targets")
        label = binary(item.get("target", item.get("label")))
        video = str(item["video"])
        if video in video_groups and video_groups[video] != group:
            raise ValueError("A video alias belongs to multiple source groups")
        video_groups[video] = group
        source_hash = item.get("video_sha256", item.get("sha256"))
        if source_hash:
            if source_hash in hash_groups and hash_groups[source_hash] != group:
                raise ValueError("Exact duplicates must share group_id")
            hash_groups[source_hash] = group
        if group in groups:
            if (groups[group]["split"], groups[group]["label"]) != (split, label):
                raise ValueError(f"Conflicting labels or cross-split source group: {group}")
            groups[group]["aliases"] = sorted(set(groups[group]["aliases"] + [video]))
        else:
            groups[group] = dict(video=video, group_id=group, split=split, label=label,
                                 aliases=[video], label_scope="whole_video",
                                 label_source=item.get("label_source", LABEL_SOURCE),
                                 window_violation_target=None)
    return sorted(groups.values(), key=lambda row: row["group_id"])


def grouped_train_partition(targets: list[dict], seed: str = "gemma-adapt-v1",
                            tune_fraction: float = 0.25) -> dict:
    """Freeze label-stratified, deterministic source-group tuning inside train."""
    if not 0 < tune_fraction < 1:
        raise ValueError("tune_fraction must be between zero and one")
    groups = video_targets(targets)
    assignments = {}
    for label in (0, 1):
        eligible = [row for row in groups if row["split"] == "train" and row["label"] == label]
        eligible.sort(key=lambda row: hashlib.sha256((seed + ":" + row["group_id"]).encode()).hexdigest())
        # Keep at least one fitting group per represented class. Singletons cannot tune.
        count = min(len(eligible) - 1, max(1, round(len(eligible) * tune_fraction))) if len(eligible) > 1 else 0
        for index, row in enumerate(eligible):
            assignments[row["group_id"]] = "train_tune" if index < count else "train_fit"
    for row in groups:
        if row["split"] != "train":
            assignments[row["group_id"]] = "reused_val_diagnostic" if row["split"] == "val" else "heldout_test"
    return dict(schema_version=VERSION, seed=seed, tune_fraction=tune_fraction,
                assignments=assignments, counts=dict(Counter(assignments.values())),
                manifest_unchanged=True, grouping="original source video, including all aliases",
                validation_caveat="Existing val has been repeatedly used; it is not an independent test.",
                parameter_selection_scope="train_tune only; fit examples and adapters use train_fit only")


def validate_fit_sources(group_ids: list[str], partition: dict) -> None:
    partition = normalize_partition(partition)
    disallowed = [group for group in group_ids if partition["assignments"].get(group) != "train_fit"]
    if disallowed:
        raise ValueError("Fit/prompt examples contain non-fit source groups: " + ", ".join(sorted(set(disallowed))))


def normalize_partition(partition: dict) -> dict:
    """Consume the audited frozen protocol without reselecting any source group."""
    if "assignments" in partition:
        return partition
    names = {"fit": "train_fit", "inner_holdout": "inner_train_holdout",
             "diagnostic_val": "reused_val_diagnostic", "test": "heldout_test"}
    assignments = {}
    for group, part in partition["group_membership"].items():
        if part not in names:
            raise ValueError("Unknown audited partition role")
        assignments[group] = names[part]
    return dict(schema_version=VERSION, seed=partition["seed"], assignments=assignments,
                counts=dict(Counter(assignments.values())), source_protocol_sha256=partition.get("protocol_sha256"),
                original_manifest_sha256=partition.get("original_manifest_sha256"),
                partition_reused_without_reselection=True,
                validation_caveat="Existing val has been repeatedly used; it is not an independent test.",
                parameter_selection_scope="Never diagnostic val; preserve the audited inner-holdout policy.")


def prediction_from_result(result: dict, group_id: str, video: str = "unknown") -> dict:
    """Read a saved blind Gemma run, never its organizer comparison or source label."""
    supported = [rule for rule in result.get("rules", [])
                 if rule.get("verdict") == "violation_supported" and rule.get("evidence")]
    # A normal episode, zero detections or all insufficient rules is NOT normal video.
    label = 1 if supported else None
    basis = "evidence_gated_rule" if supported else "abstention_no_supported_whole_video_decision"
    if result.get("whole_video_assessed") is True and result.get("whole_video_verdict") == "no_violation_observed":
        if supported:
            raise ValueError("A whole-video normal decision conflicts with supported violations")
        if result.get("status") == "completed" and result.get("coverage", {}).get("all_sampled_frames_inspected") is True:
            label = 0
            basis = "explicit_whole_video_assessment"
    return dict(group_id=group_id, video=video, predicted_label=label,
                decision_scope="whole_video", decision_basis=basis,
                status=result.get("status", "unknown"), coverage=result.get("coverage", {}),
                model_id=result.get("model_id"), model_revision=result.get("model_revision"),
                prompt_sha256=result.get("prompt_sha256"), elapsed_sec=result.get("elapsed_sec"),
                organizer_answers_in_prompt=result.get("organizer_answers_in_prompt"),
                supported_rule_ids=[rule["rule_id"] for rule in supported])


def prediction_label(row: dict) -> int | None:
    if row.get("decision_scope") != "whole_video":
        raise ValueError("Episode verdicts cannot be scored as whole-video predictions")
    if row.get("organizer_answers_in_prompt") is True:
        raise ValueError("Target-contaminated predictions must not enter blind evaluation")
    value = row.get("predicted_label")
    if value is None or isinstance(value, str) and value in ABSTAIN:
        return None
    return binary(value)


def divide(a: int, b: int) -> float | None:
    return a / b if b else None


def wilson(success: int, total: int) -> list[float] | None:
    if not total:
        return None
    z = 1.959963984540054
    p = success / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    spread = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - spread), min(1.0, center + spread)]


def classify(targets: list[dict], predictions: list[dict]) -> dict:
    labels = video_targets(targets)
    lookup = {}
    target_ids = {row["group_id"] for row in labels}
    for prediction in predictions:
        group = prediction["group_id"]
        if group not in target_ids:
            raise ValueError(f"Prediction has no target in evaluation partition: {group}")
        if group in lookup:
            raise ValueError("Multiple predictions for a source group; select one condition/run first")
        prediction_label(prediction)
        lookup[group] = prediction
    counts = Counter()
    per_video = []
    status = Counter()
    for target in labels:
        prediction = lookup.get(target["group_id"])
        value = prediction_label(prediction) if prediction else None
        actual = target["label"]
        outcome = "abstain_positive" if actual else "abstain_negative"
        if value is not None:
            outcome = "tp" if actual == value == 1 else "tn" if actual == value == 0 else "fp" if value == 1 else "fn"
        counts[outcome] += 1
        status[prediction.get("status", "unknown") if prediction else "no_run"] += 1
        per_video.append(dict(**target, predicted_label=value, outcome=outcome,
                              decision_basis=prediction.get("decision_basis") if prediction else "missing_prediction",
                              prediction_present=prediction is not None))
    tp, tn, fp, fn = (counts[name] for name in ("tp", "tn", "fp", "fn"))
    answered = tp + tn + fp + fn
    total = len(labels)
    precision = divide(tp, tp + fp)
    e2e_recall = divide(tp, tp + fn + counts["abstain_positive"])
    return dict(total_source_groups=total, predictions_present=len(lookup), answered=answered,
                abstained=total - answered, decision_coverage=divide(answered, total),
                accuracy_on_answered=divide(tp + tn, answered),
                accuracy_on_answered_wilson95=wilson(tp + tn, answered),
                correct_fraction_all_groups=divide(tp + tn, total),
                correct_fraction_all_groups_wilson95=wilson(tp + tn, total),
                precision_positive=precision, recall_positive_on_answered=divide(tp, tp + fn),
                recall_positive_end_to_end=e2e_recall,
                specificity_on_answered=divide(tn, tn + fp),
                confusion_matrix=dict(true_negative=dict(predicted_negative=tn, predicted_positive=fp,
                                                         abstained=counts["abstain_negative"]),
                                      true_positive=dict(predicted_negative=fn, predicted_positive=tp,
                                                         abstained=counts["abstain_positive"])),
                run_status_counts=dict(status), per_video=per_video,
                definitions=dict(accuracy_on_answered="Correct / answered; always read together with coverage.",
                                 correct_fraction_all_groups="Correct / all scheduled source groups; abstentions receive no credit, not a normal label.",
                                 recall_positive_end_to_end="TP / all positive source groups, including abstained positives.",
                                 precision_positive="TP / predicted positive videos; null when no positives predicted."))


def interval(item: dict) -> tuple[float, float]:
    start, end = float(item["start_sec"]), float(item["end_sec"])
    if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
        raise ValueError("Invalid event interval")
    return start, end


def temporal_iou(a: dict, b: dict) -> float:
    a0, a1 = interval(a)
    b0, b1 = interval(b)
    intersection = max(0.0, min(a1, b1) - max(a0, b0))
    return intersection / (max(a1, b1) - min(a0, b0))


def confirmed_action(item: dict) -> bool:
    """Accept the audited human adapter schema, not proposed organizer/model text."""
    return (item.get("action_confirmed") is True
            and item.get("eligible_for_action_training") is True
            and item.get("label_scope") == "observed_action"
            and item.get("review_status") not in {"proposed", "reject", "rejected", "quarantined"})


def event_reference_recovery(references: list[dict], predictions: list[dict],
                             iou_threshold: float = 0.3) -> dict:
    """Exact action types + one-to-one temporal matching on sparse confirmed refs.

    No false-positive, event mAP or full recall claims without complete reviewed
    event inventory. An accepted coarse interval is not a precisely retimed edge.
    """
    if not 0 < iou_threshold <= 1:
        raise ValueError("IoU threshold must be in (0, 1]")
    accepted, excluded = [], []
    for row in references:
        if not confirmed_action(row):
            excluded.append(dict(task_id=row.get("task_id"), reason="not_explicitly_human_confirmed_action"))
            continue
        if row.get("observed_action") in {"unclear_action", "other_observed_action", "cash_visible", "safe_open_visible"}:
            excluded.append(dict(task_id=row.get("task_id"), reason="state_or_unspecified_action"))
            continue
        interval(row)
        accepted.append(row)
    proposals = [row for row in predictions if row.get("kind", "event") == "event"]
    for row in proposals:
        interval(row)
    # Maximum-cardinality bipartite matching avoids one broad interval recovering
    # several references, and avoids a greedy match hiding a possible second match.
    edges: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for ref_index, ref in enumerate(accepted):
        for pred_index, pred in enumerate(proposals):
            if ref["group_id"] != pred["group_id"] or ref["observed_action"] != pred.get("action", pred.get("observed_action")):
                continue
            score = temporal_iou(ref, pred)
            if score >= iou_threshold:
                edges[ref_index].append((score, pred_index))
        edges[ref_index].sort(reverse=True)
    used: dict[int, int] = {}

    def augment(ref_index: int, visited: set[int]) -> bool:
        for _, pred_index in edges[ref_index]:
            if pred_index in visited:
                continue
            visited.add(pred_index)
            if pred_index not in used or augment(used[pred_index], visited):
                used[pred_index] = ref_index
                return True
        return False

    for ref_index in range(len(accepted)):
        augment(ref_index, set())
    matched = {ref_index: pred_index for pred_index, ref_index in used.items()}
    by_action = defaultdict(lambda: dict(confirmed_references=0, recovered=0))
    matches = []
    for index, ref in enumerate(accepted):
        action = ref["observed_action"]
        by_action[action]["confirmed_references"] += 1
        recovered = index in matched
        by_action[action]["recovered"] += int(recovered)
        pred = proposals[matched[index]] if recovered else None
        matches.append(dict(group_id=ref["group_id"], task_id=ref.get("task_id"), action=action,
                            recovered=recovered, matched_prediction_id=pred.get("observation_id") if pred else None,
                            temporal_iou=temporal_iou(ref, pred) if pred else None,
                            boundary_status=ref.get("boundary_status", "not_recorded")))
    return dict(confirmed_references=len(accepted), recovered=len(matched),
                sparse_reference_recovery=divide(len(matched), len(accepted)),
                iou_threshold=iou_threshold, matching="one-to-one, exact action code and original source group",
                per_action=dict(by_action), matches=matches, excluded=excluded,
                prediction_count=len(proposals), unmatched_proposals=len(proposals) - len(used),
                unmatched_proposals_are_false_positives=False,
                unsupported_metrics=["event_precision", "full_event_recall", "event_mAP", "violation_event_accuracy"],
                boundary_quality="Accepted intervals may retain proposal boundaries; not independently retimed.",
                limitation="Human action inventory is sparse. Recovery is not event recall or violation accuracy.")


def evaluate(targets: list[dict], predictions: list[dict], partition: dict | None = None) -> dict:
    targets = video_targets(targets)
    partition = normalize_partition(partition) if partition else grouped_train_partition(targets)
    expected = {row["group_id"] for row in targets}
    if set(partition["assignments"]) != expected:
        raise ValueError("Partition must cover exactly the frozen target source groups")
    for row in targets:
        assignment = partition["assignments"][row["group_id"]]
        allowed = {"train_fit", "train_tune", "inner_train_holdout"} if row["split"] == "train" else {"reused_val_diagnostic"} if row["split"] == "val" else {"heldout_test"}
        if assignment not in allowed:
            raise ValueError("Partition changes a frozen train/val/test assignment")
    for prediction in predictions:
        if prediction["group_id"] not in expected:
            raise ValueError("Unknown prediction source group")
    per_partition = {}
    for name in sorted(set(partition["assignments"].values())):
        ids = {group for group, part in partition["assignments"].items() if part == name}
        per_partition[name] = classify([row for row in targets if row["group_id"] in ids],
                                       [row for row in predictions if row["group_id"] in ids])
    return dict(schema_version=VERSION, partition=partition, whole_video=per_partition,
                all_partitions_pooled_accuracy=None,
                no_threshold_fit=True, organizer_text_inference_forbidden=True,
                limitations=["Do not pool train fitting, train tuning and reused val into one accuracy.",
                             "Original labels apply to whole videos; no window violation labels are created.",
                             "Tiny grouped samples have wide uncertainty; no 80–90% guarantee.",
                             "Source labels measure agreement with organizer labels, not independently verified legal violations."])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--partition", type=Path)
    parser.add_argument("--seed", default="gemma-adapt-v1")
    parser.add_argument("--tune-fraction", type=float, default=0.25)
    parser.add_argument("--human-events", type=Path)
    parser.add_argument("--predicted-events", type=Path)
    args = parser.parse_args()
    targets = video_targets(read_rows(args.targets))
    partition = normalize_partition(json.loads(args.partition.read_text())) if args.partition else grouped_train_partition(targets, args.seed, args.tune_fraction)
    report = evaluate(targets, read_rows(args.predictions), partition)
    report["inputs"] = {name: dict(path=str(path), sha256=digest(path)) for name, path in
                        (("targets", args.targets), ("predictions", args.predictions))}
    if bool(args.human_events) != bool(args.predicted_events):
        parser.error("Provide both --human-events and --predicted-events")
    if args.human_events:
        human, predicted = read_rows(args.human_events), read_rows(args.predicted_events)
        all_groups = {row["group_id"] for row in targets}
        if any(row["group_id"] not in all_groups for row in human + predicted):
            raise ValueError("Unknown event source group")
        report["events"] = {}
        for name in sorted(set(partition["assignments"].values())):
            groups = {group for group, part in partition["assignments"].items() if part == name}
            report["events"][name] = event_reference_recovery(
                [row for row in human if row["group_id"] in groups],
                [row for row in predicted if row["group_id"] in groups])
        for name, path in (("human_events", args.human_events), ("predicted_events", args.predicted_events)):
            report["inputs"][name] = dict(path=str(path), sha256=digest(path))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError("Choose a new output path; evaluation artifacts are append-only")
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(output=str(args.output), partitions={name: {
        key: value[key] for key in ("total_source_groups", "answered", "decision_coverage",
                                   "accuracy_on_answered", "correct_fraction_all_groups")}
        for name, value in report["whole_video"].items()}), ensure_ascii=False))


if __name__ == "__main__":
    main()
