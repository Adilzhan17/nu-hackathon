"""Immutable train-only action datasets from explicit human review decisions.

Export does not decode frames, alter source annotations, launch training, or turn
an action assessment into a binary violation target. Rejected proposals and
unreviewed intervals are never negative examples.

    python ml/export_human_dataset.py --root /root/nuhack
    python ml/export_human_dataset.py --root /root/nuhack --validate SNAPSHOT

HumanActionDataset implements the torch Dataset protocol without importing torch,
PyAV or numpy until __getitem__. It returns full-scene, aspect-preserving samples
with a dense action interval and wider temporal context, not center crops.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import uuid
import zipfile

try:
    from .human_review import ACTIONS, ACTION_OBJECTS, OBJECTS, HumanReview
except ImportError:
    from human_review import ACTIONS, ACTION_OBJECTS, OBJECTS, HumanReview


SCHEMA_VERSION = 1
TAXONOMY_VERSION = "human_observed_actions_v1"
# Class IDs must not depend on which classes happened to be reviewed this time.
# Extend through a new taxonomy version rather than reordering an existing map.
ACTION_CODES = (
    "person_enters", "person_leaves", "leave_workstation", "return_to_workstation",
    "cash_received", "cash_transferred", "cash_placed", "cash_removed", "cash_visible",
    "cash_unpacking", "cash_bundling", "cash_counter_loading", "cash_counting",
    "cash_detector_check", "cash_stored", "document_transfer", "document_handling",
    "document_checked", "document_stored", "key_handling", "key_left",
    "stamp_handling", "stamp_stored", "safe_opening", "safe_closing", "safe_locking",
    "safe_open_visible", "computer_work", "phone_used", "phone_put_away",
    "drink_at_workstation", "object_handling", "other_observed_action",
)
ACTION_TO_ID = {action: index for index, action in enumerate(ACTION_CODES)}
PROTECTED_INPUTS = (
    "data/training/manifest.csv", "data/review/event_review.csv",
    "data/transcripts/observations.jsonl", "data/transcripts/sources.json",
)
ASSESSMENTS = {"not_assessed", "normal_action", "possible_violation"}


class DatasetExportError(ValueError):
    """Unsafe, stale or malformed supervision. Nothing should be trained on it."""


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _inside(path, root):
    root = Path(root).resolve()
    path = Path(path).resolve()
    if not path.is_relative_to(root):
        raise DatasetExportError("A dataset path escapes its allowed directory")
    return path


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise DatasetExportError(f"{name} must be a finite number")
    return float(value)


def _hashes(root):
    result = {}
    for name in PROTECTED_INPUTS:
        path = _inside(root / name, root)
        if not path.is_file():
            raise DatasetExportError(f"Missing protected input: {name}")
        result[name] = _sha(path)
    return result


def _registry(root):
    """Read the original split and group assignment, never derive a new split."""
    by_video, by_hash, groups = {}, {}, {}
    with (root / PROTECTED_INPUTS[0]).open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            video, group, split, sha = (row.get(k, "") for k in ("video", "group_id", "split", "sha256"))
            if not video or not group or split not in {"train", "val", "test"} or not re.fullmatch(r"[a-f0-9]{64}", sha):
                raise DatasetExportError("Invalid source manifest identity")
            supplied = Path(video)
            if supplied.is_absolute() or not (root / "data/raw/open" / video).resolve().is_relative_to((root / "data/raw/open").resolve()):
                raise DatasetExportError("Source manifest video escapes the data directory")
            identity = {"video": video, "group_id": group, "split": split, "sha256": sha}
            if video in by_video and by_video[video] != identity:
                raise DatasetExportError("One video has inconsistent identity or split")
            if sha in by_hash and by_hash[sha] != (group, split):
                raise DatasetExportError("Identical video content leaks across groups or splits")
            if group in groups and groups[group] != split:
                raise DatasetExportError("Source group leaks across splits")
            by_video[video] = identity
            by_hash[sha] = (group, split)
            groups[group] = split
    return by_video


def _label_map():
    labels = dict(ACTIONS)
    if set(labels) - {"unclear_action"} != set(ACTION_CODES):
        raise DatasetExportError("Review taxonomy changed; version the training taxonomy before exporting")
    return {"version": TAXONOMY_VERSION, "label_scope": "observed_action",
            "action_to_id": ACTION_TO_ID,
            "actions": [{"id": ACTION_TO_ID[code], "action": code, "label": labels[code]} for code in ACTION_CODES],
            "unmentioned_actions_are_negative": False,
            "event_assessment_is_training_target": False}


def _time_string(value, name):
    if not isinstance(value, str):
        raise DatasetExportError(f"Missing {name}")
    try:
        timestamp = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise DatasetExportError(f"Invalid {name}") from exc
    if timestamp.tzinfo is None:
        raise DatasetExportError(f"{name} must include a timezone")
    return value


def _record(task, decision, registry, fingerprint):
    """Validate again at the export boundary, even when a store validated writes."""
    for key in ("video", "group_id", "split", "source_kind", "source_id"):
        if not task.get(key) or decision.get(key) != task[key]:
            raise DatasetExportError(f"Decision/task {key} mismatch")
    if task.get("id") != decision.get("task_id"):
        raise DatasetExportError("Decision/task ID mismatch")
    if task.get("source_fingerprint") != fingerprint or decision.get("source_fingerprint") != fingerprint:
        raise DatasetExportError("Decision refers to changed review inputs")
    source = registry.get(task["video"])
    if not source or source["split"] != "train" or task["split"] != "train" or source["group_id"] != task["group_id"]:
        raise DatasetExportError("Decision is outside its frozen train group")
    for value in (task.get("video_sha256"), decision.get("video_sha256")):
        if value is not None and value != source["sha256"]:
            raise DatasetExportError("Decision video SHA does not match the source manifest")
    if (decision.get("decision") != "accept" or decision.get("action_confirmed") is not True
            or decision.get("eligible_for_action_training") is not True):
        raise DatasetExportError("An explicit human action confirmation is required")
    if decision.get("violation_confirmed") is not False or decision.get("eligible_for_violation_training") is not False:
        raise DatasetExportError("Action review cannot certify a legal violation")
    if decision.get("rejected_proposal_is_negative_label") is not False:
        raise DatasetExportError("Rejected proposals must not become negative labels")
    duration = _number(task.get("duration_sec"), "duration_sec")
    start = _number(decision.get("start_sec"), "start_sec")
    end = _number(decision.get("end_sec"), "end_sec")
    evidence = _number(decision.get("evidence_time_sec"), "evidence_time_sec")
    if not 0 <= start < end <= duration or not start <= evidence <= end:
        raise DatasetExportError("Action bounds or evidence exceed the source video interval")
    action = decision.get("observed_action")
    objects = decision.get("objects")
    if (action not in ACTION_TO_ID or not isinstance(objects, list)
            or any(not isinstance(o, str) or o not in dict(OBJECTS) for o in objects)):
        raise DatasetExportError("Unknown reviewed action or object")
    if not ACTION_OBJECTS.get(action, set()).issubset(objects):
        raise DatasetExportError("The confirmed action and objects disagree")
    if decision.get("visibility") not in {"clear", "partial"}:
        raise DatasetExportError("An unobservable action cannot become a confirmed label")
    reviewer, note = decision.get("reviewer"), decision.get("note")
    if not isinstance(reviewer, str) or not reviewer.strip() or not isinstance(note, str):
        raise DatasetExportError("Human reviewer and note fields are required")
    if action == "other_observed_action" and not note.strip():
        raise DatasetExportError("Other observed action requires its human description")
    assessment = decision.get("event_assessment")
    if assessment not in ASSESSMENTS:
        raise DatasetExportError("Invalid event assessment")
    violation_type = decision.get("violation_type")
    if (assessment == "possible_violation") != bool(violation_type):
        raise DatasetExportError("Possible violation requires a separate, unverified type")
    version, revision = decision.get("version"), decision.get("revision")
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in (version, revision)):
        raise DatasetExportError("Invalid review version or revision")
    reviewed_at = _time_string(decision.get("updated_at"), "updated_at")
    created_at = _time_string(decision.get("created_at"), "created_at")
    return {
        "schema_version": SCHEMA_VERSION, "task_id": task["id"], "task_version": version,
        "decision_revision": revision, "source_video": source["video"], "video": source["video"],
        "source_path": (Path("data/raw/open") / source["video"]).as_posix(),
        "video_sha256": source["sha256"], "group_id": source["group_id"], "split": "train",
        "duration_sec": duration, "start_sec": start, "end_sec": end,
        "observed_action": action, "action_id": ACTION_TO_ID[action], "objects": sorted(set(objects)),
        "label_scope": "observed_action", "label_origin": "explicit_human_review",
        "reviewer": reviewer.strip(), "reviewed_at": reviewed_at, "created_at": created_at,
        "note": note, "evidence_time_sec": evidence, "visibility": decision["visibility"],
        "event_assessment": assessment, "possible_violation_type": violation_type,
        "action_confirmed": True, "violation_confirmed": False,
        "eligible_for_action_training": True, "eligible_for_violation_training": False,
        "source_fingerprint": fingerprint,
        "assessment_authority": "human_review", "normal_action_is_whole_video_label": False,
        "provenance": {"source_kind": task["source_kind"], "source_id": task["source_id"],
                       "decision_sha256": hashlib.sha256(_json(decision).encode()).hexdigest()},
    }


def _deduplicate(records):
    """Merge only exact duplicate action annotations, retaining every reviewer."""
    groups = {}
    for record in records:
        key = _json([record[k] for k in ("video_sha256", "observed_action", "objects", "start_sec", "end_sec", "event_assessment")])
        groups.setdefault(key, []).append(record)
    merged = []
    for group in groups.values():
        ordered = sorted(group, key=lambda row: row["task_id"])
        row = dict(ordered[0])
        row["source_task_ids"] = [r["task_id"] for r in ordered]
        row["review_audit"] = [{k: r[k] for k in (
            "task_id", "task_version", "decision_revision", "reviewer", "reviewed_at", "created_at",
            "evidence_time_sec", "visibility", "note", "possible_violation_type", "provenance")}
            for r in ordered]
        merged.append(row)
    merged.sort(key=lambda row: row["task_id"])
    per_group = Counter(row["group_id"] for row in merged)
    for row in merged:
        row["group_balanced_weight"] = 1.0 / per_group[row["group_id"]]
    return merged


def _overlap_counts(rows):
    overlaps = same_action = 0
    for i, row in enumerate(rows):
        for other in rows[i + 1:]:
            if (row["video_sha256"] == other["video_sha256"]
                    and max(row["start_sec"], other["start_sec"]) < min(row["end_sec"], other["end_sec"])):
                overlaps += 1
                same_action += row["observed_action"] == other["observed_action"]
    return {"overlapping_annotation_pairs": overlaps, "same_action_overlapping_pairs": same_action,
            "approximate_overlaps_merged": False}


def _write(path, value):
    with path.open("x", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def export_snapshot(root, store=None, snapshot_id=None):
    """Create a new immutable snapshot; returns metadata, never starts training.

    A concurrent edit to an included review aborts the export. Unrelated new
    decisions may be added while this consistent snapshot is being written.
    """
    root = Path(root).resolve()
    store = store if store is not None else HumanReview(root)
    before = _hashes(root)
    catalog = store.catalog()
    fingerprint = catalog.get("source_fingerprint")
    if not isinstance(fingerprint, str) or not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
        raise DatasetExportError("Missing review source fingerprint")
    registry = _registry(root)
    label_map = _label_map()
    task_list = catalog.get("tasks", [])
    tasks = {t["id"]: t for t in task_list}
    if len(tasks) != len(task_list):
        raise DatasetExportError("Duplicate review task IDs")
    decisions = catalog.get("decisions", {})
    rows, excluded = [], Counter()
    for task_id, decision in sorted(decisions.items()):
        task = tasks.get(task_id)
        if not task or decision.get("source_fingerprint") != fingerprint:
            excluded["stale_or_removed_task"] += 1
            continue
        if decision.get("decision") in {"reject", "needs_context"}:
            excluded[decision["decision"]] += 1
            continue
        if decision.get("decision") != "accept":
            raise DatasetExportError("Unrecognized review decision")
        rows.append(_record(task, decision, registry, fingerprint))
    accepted_decisions = len(rows)
    rows = _deduplicate(rows)
    verified_paths = set()
    for row in rows:
        path = _inside(root / row["source_path"], root / "data/raw/open")
        if path not in verified_paths:
            if not path.is_file() or _sha(path) != row["video_sha256"]:
                raise DatasetExportError("Source video content changed or is missing")
            verified_paths.add(path)
    now = dt.datetime.now(dt.timezone.utc)
    identifier = snapshot_id or f"human_actions_{now.strftime('%Y%m%dT%H%M%S%fZ')}_{uuid.uuid4().hex[:8]}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}", identifier):
        raise DatasetExportError("Invalid snapshot ID")
    exports = _inside(root / "data/human_review/exports", root / "data/human_review")
    exports.mkdir(parents=True, exist_ok=True)
    destination = exports / identifier
    with (exports / ".export.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if destination.exists():
            raise FileExistsError("A snapshot is immutable; choose a new snapshot ID")
        temporary = Path(tempfile.mkdtemp(prefix=f".{identifier}.", dir=exports))
        try:
            _write(temporary / "manifest.jsonl", "".join(_json(row) + "\n" for row in rows))
            _write(temporary / "label_map.json", _json(label_map) + "\n")
            current = store.catalog()
            if before != _hashes(root) or current.get("source_fingerprint") != fingerprint:
                raise DatasetExportError("Review source files changed during export")
            for row in rows:
                for audit in row["review_audit"]:
                    latest = current.get("decisions", {}).get(audit["task_id"])
                    if latest is None or hashlib.sha256(_json(latest).encode()).hexdigest() != audit["provenance"]["decision_sha256"]:
                        raise DatasetExportError("An included review changed during export; retry")
            metadata = {
                "schema_version": SCHEMA_VERSION, "snapshot_id": identifier, "export_id": identifier,
                "created_at": now.isoformat(timespec="microseconds"),
                "path": str(destination), "relative_path": destination.relative_to(root).as_posix(),
                "manifest": "manifest.jsonl", "taxonomy": "label_map.json",
                "archive_relative_path": (destination.relative_to(root) / "dataset.zip").as_posix(),
                "taxonomy_version": TAXONOMY_VERSION,
                "status": "ready_for_action_training" if rows else "empty_waiting_for_human_review",
                "count": len(rows), "training_actions": len(rows), "has_training_examples": bool(rows), "training_started": False,
                "accepted_review_decisions": accepted_decisions,
                "exact_duplicate_annotations_merged": accepted_decisions - len(rows),
                "overlap_checks": _overlap_counts(rows),
                "unique_source_videos": len({r["video_sha256"] for r in rows}),
                "unique_groups": len({r["group_id"] for r in rows}),
                "source_video_groups": len({r["group_id"] for r in rows}),
                "sampling_weight": "group_balanced_weight sums to one per source group; reviewed intervals are correlated",
                "required_training_inputs": "Original videos, protected source annotations, review reports and current human review store must accompany this metadata-only archive",
                "action_counts": dict(Counter(r["observed_action"] for r in rows)),
                "event_assessment_counts": dict(Counter(r["event_assessment"] for r in rows)),
                "excluded_decisions": dict(excluded), "unreviewed_tasks": len(set(tasks) - set(decisions)),
                "review_revision": catalog.get("revision"), "source_fingerprint": fingerprint,
                "protected_inputs": before,
                "artifact_sha256": {name: _sha(temporary / name) for name in ("manifest.jsonl", "label_map.json")},
                "training_objective": "observed_action_recognition",
                "label_scope": "observed_action", "split": "train",
                "violation_training_supported": False,
                "policy": {
                    "normal_action_is_whole_video_normal": False,
                    "possible_violation_is_verified_legal_violation": False,
                    "reject_or_needs_context_is_negative_label": False,
                    "unannotated_interval_is_negative_label": False,
                    "overlapping_actions_are_allowed": True,
                    "held_out_videos_exported": False,
                    "original_annotations_modified": False,
                    "frames_exported": False,
                },
            }
            _write(temporary / "metadata.json", _json(metadata) + "\n")
            with zipfile.ZipFile(temporary / "dataset.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
                for name in ("manifest.jsonl", "label_map.json", "metadata.json"):
                    archive.write(temporary / name, arcname=name)
            with (temporary / "dataset.zip").open("rb") as handle:
                os.fsync(handle.fileno())
            _fsync_directory(temporary)
            os.rename(temporary, destination)
            _fsync_directory(exports)
            return metadata
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)


def _load_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_snapshot(snapshot, root, require_nonempty=False, store=None, verify_video_hashes=True):
    """Reject edited artifacts, changed sources, split leakage and retracted labels.

    New unrelated reviews do not invalidate a frozen snapshot. An edited or
    retracted included review does: exporting again is then mandatory.
    """
    root = Path(root).resolve()
    snapshot = Path(snapshot)
    if not snapshot.is_absolute():
        snapshot = root / snapshot
    snapshot = _inside(snapshot, root / "data/human_review/exports")
    meta = json.loads((snapshot / "metadata.json").read_text(encoding="utf-8"))
    if meta.get("schema_version") != SCHEMA_VERSION or meta.get("taxonomy_version") != TAXONOMY_VERSION:
        raise DatasetExportError("Unsupported human dataset version")
    if meta.get("label_scope") != "observed_action" or meta.get("split") != "train" or meta.get("violation_training_supported") is not False:
        raise DatasetExportError("Unsupported training target or split")
    if meta.get("protected_inputs") != _hashes(root):
        raise DatasetExportError("Protected source annotations changed after export")
    expected_artifacts = {"manifest.jsonl", "label_map.json"}
    if set(meta.get("artifact_sha256", {})) != expected_artifacts:
        raise DatasetExportError("Missing snapshot artifact checksums")
    for name, digest in meta["artifact_sha256"].items():
        artifact = _inside(snapshot / name, snapshot)
        if not artifact.is_file() or _sha(artifact) != digest:
            raise DatasetExportError("Snapshot artifact content changed")
    try:
        with zipfile.ZipFile(_inside(snapshot / "dataset.zip", snapshot)) as archive:
            expected_names = expected_artifacts | {"metadata.json"}
            if len(archive.namelist()) != len(expected_names) or set(archive.namelist()) != expected_names:
                raise DatasetExportError("Snapshot archive contains unexpected files")
            if any(archive.read(name) != (snapshot / name).read_bytes() for name in expected_names):
                raise DatasetExportError("Snapshot archive differs from its dataset files")
    except zipfile.BadZipFile as exc:
        raise DatasetExportError("Snapshot archive is corrupt") from exc
    label_map = json.loads((snapshot / "label_map.json").read_text(encoding="utf-8"))
    if label_map.get("version") != TAXONOMY_VERSION or label_map.get("action_to_id") != ACTION_TO_ID:
        raise DatasetExportError("Action IDs changed")
    rows = _load_jsonl(snapshot / "manifest.jsonl")
    if len(rows) != meta.get("count") or len({r["task_id"] for r in rows}) != len(rows):
        raise DatasetExportError("Invalid dataset count or duplicate task IDs")
    if require_nonempty and not rows:
        raise DatasetExportError("No confirmed actions yet; an empty snapshot cannot train a model")
    store = store if store is not None else HumanReview(root)
    catalog = store.catalog()
    if catalog.get("source_fingerprint") != meta["source_fingerprint"]:
        raise DatasetExportError("Review context changed after export")
    registry, tasks, checked = _registry(root), {t["id"]: t for t in catalog["tasks"]}, set()
    current_records, included_ids = [], set()
    for row in rows:
        source_ids = row.get("source_task_ids", [])
        if not source_ids or len(source_ids) != len(set(source_ids)) or included_ids.intersection(source_ids):
            raise DatasetExportError("Missing or duplicated merged review provenance")
        included_ids.update(source_ids)
        for task_id in source_ids:
            decision = catalog.get("decisions", {}).get(task_id)
            task = tasks.get(task_id)
            if not decision or not task or decision.get("decision") != "accept":
                raise DatasetExportError("An included human label was removed or retracted")
            current_records.append(_record(task, decision, registry, meta["source_fingerprint"]))
        path = _inside(root / row["source_path"], root / "data/raw/open")
        if path not in checked:
            if not path.is_file() or (verify_video_hashes and _sha(path) != row["video_sha256"]):
                raise DatasetExportError("Source video changed after export")
            checked.add(path)
    if rows != _deduplicate(current_records):
        raise DatasetExportError("An included human review changed after export")
    return {"valid": True, "count": len(rows), "has_training_examples": bool(rows),
            "snapshot_id": meta["snapshot_id"], "label_scope": "observed_action",
            "video_hashes_checked": bool(verify_video_hashes), "metadata": meta}


class HumanActionDataset:
    """Torch-compatible samples with [C,T,H,W] float RGB tensors in [0,1].

    No normalization is assumed: apply the selected model's transform explicitly.
    Half of the frames cover the surrounding context, half cover the reviewed
    action. The focus mask marks source timestamps inside that action. Samples
    carry action IDs only, never binary normal/violation targets. All return
    fields can be collated by torch's default DataLoader.
    """
    def __init__(self, root, snapshot, num_frames=16, size=384, context_seconds=4.0, transform=None, store=None):
        self.root = Path(root).resolve()
        self.snapshot = Path(snapshot)
        if not self.snapshot.is_absolute():
            self.snapshot = self.root / self.snapshot
        if (isinstance(num_frames, bool) or not isinstance(num_frames, int) or num_frames < 4
                or isinstance(size, bool) or not isinstance(size, int) or size < 384):
            raise DatasetExportError("Use at least four frames and a size of at least 384 pixels")
        self.context_seconds = _number(context_seconds, "context_seconds")
        if not 0 <= self.context_seconds <= 60:
            raise DatasetExportError("Context must be between 0 and 60 seconds on either side")
        validate_snapshot(self.snapshot, self.root, require_nonempty=True, store=store)
        self.rows = _load_jsonl(self.snapshot / "manifest.jsonl")
        self.num_frames, self.size, self.transform = num_frames, size, transform
        self.label_map = dict(ACTION_TO_ID)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        import av
        import numpy as np
        import torch

        row = self.rows[index]
        start, end, duration = row["start_sec"], row["end_sec"], row["duration_sec"]
        context_start = max(0.0, start - self.context_seconds)
        context_end = min(duration, end + self.context_seconds)
        def times(a, b, n):
            margin = min(0.04, (b - a) / (2 * max(n, 1)))
            return np.linspace(a + margin, b - margin, n)
        if self.context_seconds:
            count = self.num_frames // 2
            targets = np.sort(np.concatenate((times(context_start, context_end, count), times(start, end, self.num_frames - count))))
        else:
            targets = times(start, end, self.num_frames)
        path = _inside(self.root / row["source_path"], self.root / "data/raw/open")
        images, timestamps = [], []
        def append_frame(chosen, chosen_time):
            scale = self.size / max(chosen.width, chosen.height)
            array = chosen.reformat(width=max(1, min(self.size, round(chosen.width * scale))),
                                    height=max(1, min(self.size, round(chosen.height * scale))), format="rgb24").to_ndarray()
            canvas = np.full((self.size, self.size, 3), 114, dtype=np.uint8)
            y, x = (self.size - array.shape[0]) // 2, (self.size - array.shape[1]) // 2
            canvas[y:y + array.shape[0], x:x + array.shape[1]] = array
            images.append(canvas)
            timestamps.append(chosen_time)
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            stream.codec_context.thread_count = 2
            origin = float((stream.start_time or 0) * stream.time_base)
            container.seek(int((context_start + origin) / float(stream.time_base)), stream=stream, backward=True)
            previous = None
            previous_time = None
            for frame in container.decode(video=0):
                if frame.pts is None:
                    continue
                timestamp = float(frame.pts * stream.time_base) - origin
                if timestamp < 0:
                    continue
                while len(images) < len(targets) and timestamp >= targets[len(images)]:
                    wanted = targets[len(images)]
                    chosen, chosen_time = frame, timestamp
                    if previous is not None and abs(previous_time - wanted) <= abs(timestamp - wanted):
                        chosen, chosen_time = previous, previous_time
                    append_frame(chosen, chosen_time)
                previous, previous_time = frame, timestamp
                if len(images) == self.num_frames:
                    break
            # The last presentation timestamp is normally one frame before the
            # container duration. Nearest-frame sampling may legitimately select
            # it; a larger missing tail remains an error rather than being padded.
            frame_step = 1.0 / float(stream.average_rate) if stream.average_rate else 0.04
            if (len(images) < self.num_frames and previous is not None
                    and 0 <= targets[-1] - previous_time <= frame_step + 1e-3):
                while len(images) < self.num_frames:
                    append_frame(previous, previous_time)
        if len(images) != self.num_frames:
            raise DatasetExportError("Source video ended before the requested action/context frames; check duration and decoding")
        tensor = torch.from_numpy(np.stack(images)).permute(3, 0, 1, 2).float().div_(255)
        if self.transform is not None:
            tensor = self.transform(tensor)
        times_tensor = torch.tensor(timestamps, dtype=torch.float64)
        focus_mask = (times_tensor >= start) & (times_tensor <= end)
        if not bool(focus_mask.any()):
            raise DatasetExportError("No sampled source frame falls inside the confirmed action interval")
        return {"pixel_values": tensor, "action_id": torch.tensor(row["action_id"], dtype=torch.long),
                "frame_timestamps": times_tensor, "focus_mask": focus_mask,
                "action_bounds": torch.tensor([start, end], dtype=torch.float64),
                "context_bounds": torch.tensor([context_start, context_end], dtype=torch.float64),
                "group_balanced_weight": torch.tensor(row["group_balanced_weight"], dtype=torch.float32),
                "task_id": row["task_id"], "observed_action": row["observed_action"],
                "event_assessment": row["event_assessment"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--snapshot-id")
    parser.add_argument("--validate", type=Path)
    parser.add_argument("--require-nonempty", action="store_true")
    args = parser.parse_args()
    try:
        result = (validate_snapshot(args.validate, args.root, require_nonempty=args.require_nonempty)
                  if args.validate else export_snapshot(args.root, snapshot_id=args.snapshot_id))
        # Metadata contains counts and provenance hashes, not video descriptions or frames.
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (DatasetExportError, FileExistsError) as exc:
        print(json.dumps({"valid": False, "error": str(exc)}, ensure_ascii=False))
        raise SystemExit(2)


if __name__ == "__main__":
    main()
