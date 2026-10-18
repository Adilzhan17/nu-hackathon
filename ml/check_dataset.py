"""Validate explicit clip annotations before spending GPU time.

Usage: python3 ml/check_dataset.py annotations.csv --root /path/to/videos
CSV columns: video,group_id,split,start_sec,end_sec,label
split: train/val/test. label: normal or a manually reviewed event type.
group_id must group related takes/scenes, not just individual extracted clips.
All paths are relative to --root. Unannotated intervals are NOT assumed normal.
This checks annotation structure and leakage, not video decodability or label quality.
"""

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def validate(manifest: Path, root: Path, allow_weak: bool = False) -> dict:
    root = root.resolve()
    errors = []
    counts = defaultdict(Counter)
    group_splits = defaultdict(set)
    file_splits = defaultdict(set)
    file_groups = defaultdict(set)
    hashes = {}
    duplicates = defaultdict(list)
    intervals = defaultdict(list)
    annotation_modes = Counter()
    binary_events = 0
    required = {"video", "group_id", "split", "start_sec", "end_sec", "label"}
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not required.issubset(reader.fieldnames or []):
            return {"valid": False, "errors": [f"Required columns: {sorted(required)}"]}
        for line, row in enumerate(reader, 2):
            if None in row or any(row.get(key) is None for key in required):
                errors.append(f"Line {line}: malformed CSV row")
                continue
            row = {key: (value or "").strip() for key, value in row.items()}
            accepted = {"reviewed", "source_labeled"} if allow_weak else {"reviewed"}
            if "review_status" in row and row["review_status"] not in accepted:
                errors.append(f"Line {line}: human review required before training")
                continue
            mode = row.get("annotation_mode") or "exact"
            if mode not in {"exact", "point_window", "source_video_label"}:
                errors.append(f"Line {line}: unknown annotation mode")
                continue
            if mode != "exact" and not allow_weak:
                errors.append(f"Line {line}: approximate/source labels require --allow-weak")
                continue
            annotation_modes[mode] += 1
            if row["label"] == "violation_unspecified":
                binary_events += 1
            split, label, group = row["split"], row["label"], row["group_id"]
            if split not in {"train", "val", "test"} or not label or not group:
                errors.append(f"Line {line}: invalid split, empty label or group_id")
                continue
            try:
                start, end = float(row["start_sec"]), float(row["end_sec"])
                if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
                    raise ValueError
            except ValueError:
                errors.append(f"Line {line}: require finite 0 <= start_sec < end_sec")
                continue
            supplied = Path(row["video"])
            path = (root / supplied).resolve()
            if supplied.is_absolute() or not path.is_relative_to(root) or not path.is_file():
                errors.append(f"Line {line}: video must be an existing file inside --root")
                continue
            if path.suffix.lower() not in {".mov", ".mp4", ".mkv", ".avi"} or path.stat().st_size == 0:
                errors.append(f"Line {line}: empty file or unsupported video extension")
                continue
            if path not in hashes:
                with path.open("rb") as video:
                    hashes[path] = hashlib.file_digest(video, "sha256").hexdigest()
            digest = hashes[path]
            group_splits[group].add(split)
            file_splits[digest].add(split)
            file_groups[digest].add(group)
            duplicates[digest].append(row["video"])
            intervals[digest].append((start, end, label, line, mode))
            counts[split][label] += 1
    for group, splits in group_splits.items():
        if len(splits) > 1:
            errors.append(f"Group {group!r} leaks across splits: {sorted(splits)}")
    for digest, splits in file_splits.items():
        names = sorted(set(duplicates[digest]))
        if len(splits) > 1:
            errors.append(f"Same video content leaks across splits: {names}")
        if len(file_groups[digest]) > 1:
            errors.append(f"Same video content assigned to different groups: {names}")
        rows = intervals[digest]
        for i, (start, end, label, line, mode) in enumerate(rows):
            for other_start, other_end, other_label, other_line, other_mode in rows[i + 1:]:
                if max(start, other_start) < min(end, other_end) and label != other_label:
                    if allow_weak and "point_window" in {mode, other_mode} and "normal" not in {label, other_label}:
                        continue  # Point windows can overlap: these are not exact action boundaries.
                    errors.append(
                        f"Lines {line}/{other_line}: overlapping different labels; "
                        "resolve boundaries or use a multilabel training pipeline"
                    )
    for split in ("train", "val"):
        if not counts[split]:
            errors.append(f"Missing {split} examples")
        elif "normal" not in counts[split] or len(counts[split]) < 2:
            errors.append(f"{split} needs normal examples and at least one event type")
    unseen = set(counts["val"]) - set(counts["train"])
    if unseen:
        errors.append(f"Validation classes absent from training: {sorted(unseen)}")
    missing_val = set(counts["train"]) - set(counts["val"])
    warnings = [f"No validation examples for: {sorted(missing_val)}"] if missing_val else []
    if any(mode != "exact" for mode in annotation_modes):
        warnings.append("Approximate/source labels: do not use interval boundaries as localization ground truth")
    if binary_events:
        warnings.append(f"{binary_events} unspecified event clips are binary labels, not a learned event type")
    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "annotation_modes": dict(annotation_modes),
        "clips_per_class": {split: dict(value) for split, value in counts.items()},
        "unique_video_contents": len(file_splits),
        "groups": len(group_splits),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "checks_not_performed": ["video decoding", "duration bounds", "human label review"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--allow-weak", action="store_true", help="Explicitly allow source labels and approximate point windows")
    args = parser.parse_args()
    print(json.dumps(report := validate(args.manifest, args.root, args.allow_weak), ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["valid"] else 1)
