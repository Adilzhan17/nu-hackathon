"""Versioned 95/5 video-group split without modifying the original manifest.

The two old validation videos are a diagnostic set, not an independent test.
No model predictions, scores, captions or private frames are used for selection.

activate_split writes the next-run policy consumed by train_suite.py when its
--manifest argument is omitted. An explicit --manifest overrides that policy.
The separate narration-training contract and the old train_x3d.py/dashboard
training command are not rewritten or automatically migrated by this module.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import datetime as dt
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import tempfile


VERSION = "split95_v1"
SEED = "nuhack-grouped-95-5-v1"
SOURCE = "data/training/manifest.csv"


class SplitError(ValueError):
    pass


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"


def _version_dir(root, version):
    if not re.fullmatch(r"split95_v[1-9][0-9]*", version):
        raise SplitError("Invalid split version")
    return Path(root).resolve() / "data/training" / version


def _csv(rows, fields):
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue()


def _entities(rows):
    """A source SHA is indivisible, including aliases with different group IDs."""
    groups, videos, entities = {}, {}, {}
    for row in rows:
        video, group, sha, source_split = (row.get(k, "") for k in ("video", "group_id", "sha256", "split"))
        if not video or not group or not re.fullmatch(r"[a-f0-9]{64}", sha):
            raise SplitError("Source manifest has invalid video/group/SHA identity")
        if Path(video).is_absolute() or ".." in Path(video).parts:
            raise SplitError("Video path must be relative and remain inside the source directory")
        if source_split not in {"train", "val"} or row.get("target") not in {"0", "1"}:
            raise SplitError("Expected source train/val split and explicit binary target")
        target = int(row["target"])
        if group in groups and groups[group] != sha:
            raise SplitError("A source group contains different video hashes")
        if video in videos and videos[video] != sha:
            raise SplitError("A video alias has inconsistent SHA identity")
        groups[group], videos[video] = sha, sha
        entity = entities.setdefault(sha, {"sha256": sha, "group_ids": set(), "videos": set(),
            "source_splits": set(), "targets": set(), "source_rows": 0})
        entity["group_ids"].add(group)
        entity["videos"].add(video)
        entity["source_splits"].add(source_split)
        entity["targets"].add(target)
        entity["source_rows"] += 1
    if not entities:
        raise SplitError("Source manifest is empty")
    result = []
    for sha, entity in sorted(entities.items()):
        if len(entity["source_splits"]) != 1:
            raise SplitError("Identical source bytes already leak across train/val")
        if len(entity["targets"]) != 1:
            raise SplitError("A source video has conflicting binary labels")
        result.append({"sha256": sha, "group_ids": sorted(entity["group_ids"]), "videos": sorted(entity["videos"]),
            "source_split": next(iter(entity["source_splits"])), "source_target": next(iter(entity["targets"])),
            "source_rows": entity["source_rows"]})
    return result


def build_report(rows, source_sha256, *, seed=SEED, version=VERSION):
    """Pure split selection. One old-val video per label, selected by seeded SHA rank."""
    entities = _entities(rows)
    if len(entities) != 40:
        raise SplitError("This frozen 95/5 plan requires exactly 40 distinct source video hashes")
    held_out = []
    for target in (0, 1):
        candidates = [r for r in entities if r["source_split"] == "val" and r["source_target"] == target]
        if not candidates:
            raise SplitError("The original validation set must contain both labels")
        candidate = min(candidates, key=lambda r: (_sha(f"{seed}:{target}:{r['sha256']}".encode()), r["sha256"]))
        held_out.append(candidate["sha256"])
    for entity in entities:
        entity["split"] = "diagnostic_test" if entity["sha256"] in held_out else "train"
        entity["selection_rank_sha256"] = _sha(f"{seed}:{entity['source_target']}:{entity['sha256']}".encode())
    membership = {group: entity["split"] for entity in entities for group in entity["group_ids"]}
    sha_membership = {entity["sha256"]: entity["split"] for entity in entities}
    counts = {}
    for split in ("train", "diagnostic_test"):
        selected = [r for r in entities if r["split"] == split]
        counts[split] = {"unique_videos": len(selected), "source_groups": sum(len(r["group_ids"]) for r in selected),
            "windows": sum(r["source_rows"] for r in selected),
            "video_labels": {"normal": sum(r["source_target"] == 0 for r in selected),
                             "positive": sum(r["source_target"] == 1 for r in selected)}}
    return {"schema_version": 1, "version": version, "seed": seed,
        "status": "ready_for_next_training", "training_started": False,
        "manifest_relative_path": f"data/training/{version}/manifest.csv",
        "legacy_manifest_relative_path": f"data/training/{version}/manifest_legacy.csv",
        "source_manifest": SOURCE, "source_manifest_sha256": source_sha256,
        "selection": "one original-val video of each source target; minimum SHA256(seed:target:video_sha256)",
        "selection_uses_model_metrics": False, "split_unit": "source_video_sha256_with_all_group_and_path_aliases",
        "total_unique_videos": len(entities), "total_source_groups": len(membership), "total_windows": len(rows),
        "train_fraction": .95, "diagnostic_test_fraction": .05, "counts": counts,
        "independent_test": False, "diagnostic_only": True,
        "old_validation_reused": True, "accuracy_resolution_percentage_points": 50,
        "promotion_from_old_validation": [r["sha256"] for r in entities if r["source_split"] == "val" and r["split"] == "train"],
        "group_membership": membership, "sha256_membership": sha_membership, "entities": entities,
        "policy": {"original_manifest_overwritten": False, "frames_or_captions_used": False,
            "select_checkpoint_on_diagnostic_test": False, "fit_threshold_on_diagnostic_test": False,
            "legacy_val_is_independent_validation": False,
            "training_selection": "use grouped internal cross-validation inside the 38 train source videos",
            "unannotated_intervals_are_negative": False,
            "all_windows_and_aliases_of_one_video_share_split": True},
        "limitations": ["Two previously inspected videos are not an independent benchmark.",
            "The 5% diagnostic set cannot establish 85–90% or 95% generalization quality.",
            "Previous eight-video validation scores are not comparable with the new two-video diagnostic result.",
            "Changing the split does not add confirmed actions, boxes, or violation labels."]}


def _identity(row, report):
    group = row.get("group_id")
    sha = row.get("sha256") or row.get("video_sha256") or row.get("source_sha256")
    entity = next((r for r in report["entities"] if group in r["group_ids"]), None)
    if entity is None or sha != entity["sha256"]:
        raise SplitError("Example is outside the frozen group/SHA membership")
    video = row.get("video")
    if video is not None and video not in entity["videos"]:
        raise SplitError("Example video alias disagrees with frozen group/SHA identity")
    return entity


def apply_split(rows, report, *, compatibility=False):
    """Return copies with new split membership, preserving all other manifest fields."""
    output = []
    for row in rows:
        entity = _identity(row, report)
        split = entity["split"]
        output.append(dict(row, split="val" if compatibility and split == "diagnostic_test" else split))
    return output


def filter_human_examples(rows, report, *, partition="train"):
    """Restrict an already human-approved export by identity; never invent labels.

    Returns (copied_rows, filter_statistics). Approval eligibility remains the
    responsibility of export_human_dataset; this helper only enforces membership.
    """
    if partition not in {"train", "diagnostic_test"}:
        raise SplitError("Unknown target partition")
    output, excluded, seen = [], Counter(), 0
    for row in rows:
        seen += 1
        entity = _identity(row, report)
        if entity["split"] != partition:
            excluded[entity["split"]] += 1
            continue
        copied = dict(row, split=partition, split_version=report["version"], original_example_split=row.get("split"),
                      split_source_manifest_sha256=report["source_manifest_sha256"])
        output.append(copied)
    return output, {"input_examples": seen, "included_examples": len(output), "excluded_partitions": dict(excluded),
        "partition": partition, "independent_test": False, "labels_created": 0}


def _instructions(report):
    return f"""Версия: {report['version']}

38 исходных видео для обучения; 2 видео для диагностической проверки (одно с
исходной положительной меткой, одно нормальное). Это 95/5 по уникальным видео,
а не по кадрам или окнам. Все окна и копии одного видео имеют одну часть.

Исходный data/training/manifest.csv не изменён.
manifest.csv: split=train либо diagnostic_test.
manifest_legacy.csv: split=train либо val для старых загрузчиков.
В legacy-файле val означает только диагностическую часть, а не независимую
валидацию. Не выбирайте по ней эпоху, порог или лучшую модель. Для выбора внутри
train используйте разбиение по группам. Сравните diagnostic_test один раз после
фиксации конфигурации. Все восемь прежних val-видео уже использовались при
разработке, поэтому независимого test-набора здесь нет.

Два ролика дают шаг accuracy в 50 процентных пунктов. Такой результат не докажет
качество 85–90% или 95%. Новая доля обучения сама по себе не улучшает разметку.

Подготовить / проверить эту версию:
  .venv/bin/python ml/split95.py --root /root/nuhack --activate
  .venv/bin/python ml/split95.py --root /root/nuhack --check

Запустить НОВЫЙ GPU-прогон фиксированных эпох (из корня проекта, после переноса
версии split95_v1 и исходных видео на GPU; каталог вывода должен быть новым):
  .venv-training/bin/python ml/train_suite.py --models videomaev2 --manifest data/training/{report['version']}/manifest.csv --diagnostic-test --epochs 120 --output runs/split95_v1_fixed120

Режим --diagnostic-test обучает фиксированное число эпох и оценивает замороженный
итоговый checkpoint один раз. Метки diagnostic_test не выбирают эпоху или порог.
Метрики diagnostic_test.json относятся к исходным размеченным окнам, не ко всему
пользовательскому сценарию. Запуск не происходит автоматически от --activate.

Использование Python (PYTHONPATH=ml):
  from split95 import load_split, load_manifest, filter_human_examples
  report = load_split('/root/nuhack')
  rows = load_manifest('/root/nuhack')
  train_actions, audit = filter_human_examples(approved_actions, report)

filter_human_examples проверяет group_id и SHA исходного видео; она не заменяет
экспорт подтверждённых человеком действий и не считает пропуски отрицательными
примерами. Уже утверждённые датасеты не перезаписываются. Эта команда не запускает
обучение и не подключается к GPU.
"""


def prepare(root, *, version=VERSION, seed=SEED):
    root = Path(root).resolve()
    destination = _version_dir(root, version)
    if destination.exists():
        report = load_split(root, version=version)
        if report["seed"] != seed:
            raise SplitError("A split version is immutable; choose a new version for another seed")
        return report
    raw = (root / SOURCE).read_bytes()
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
    rows = list(reader)
    fields = reader.fieldnames
    report = build_report(rows, _sha(raw), seed=seed, version=version)
    normal = _csv(apply_split(rows, report), fields)
    legacy = _csv(apply_split(rows, report, compatibility=True), fields)
    group_lines = "".join(json.dumps(entity, ensure_ascii=False, sort_keys=True) + "\n" for entity in report["entities"])
    artifacts = {"manifest.csv": normal, "manifest_legacy.csv": legacy, "groups.jsonl": group_lines,
                 "README.txt": _instructions(report)}
    report["prepared_at"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    report["artifact_sha256"] = {name: _sha(content.encode()) for name, content in artifacts.items()}
    artifacts["report.json"] = _json(report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{version}.", dir=destination.parent))
    try:
        for name, content in artifacts.items():
            path = temporary / name
            with path.open("w", encoding="utf-8", newline="") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        if _sha((root / SOURCE).read_bytes()) != report["source_manifest_sha256"]:
            raise SplitError("Original manifest changed while preparing the split")
        os.rename(temporary, destination)
        return report
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def load_split(root, *, version=VERSION):
    """Read and verify original source, artifact hashes, and deterministic membership."""
    root = Path(root).resolve()
    directory = _version_dir(root, version)
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    raw = (root / SOURCE).read_bytes()
    if _sha(raw) != report.get("source_manifest_sha256"):
        raise SplitError("Original source manifest changed after split preparation")
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
    expected = build_report(rows, _sha(raw), seed=report["seed"], version=version)
    for key, value in expected.items():
        if report.get(key) != value:
            raise SplitError(f"Frozen split metadata differs from deterministic source: {key}")
    expected_names = {"manifest.csv", "manifest_legacy.csv", "groups.jsonl", "README.txt"}
    if set(report.get("artifact_sha256", {})) != expected_names:
        raise SplitError("Unexpected or missing split artifacts")
    for name, sha in report["artifact_sha256"].items():
        if _sha((directory / name).read_bytes()) != sha:
            raise SplitError(f"Frozen split artifact was modified: {name}")
    for name, compatibility in (("manifest.csv", False), ("manifest_legacy.csv", True)):
        actual = list(csv.DictReader(io.StringIO((directory / name).read_text(encoding="utf-8"))))
        if actual != apply_split(rows, report, compatibility=compatibility):
            raise SplitError("Split manifest content differs from frozen source rows")
    return report


def load_manifest(root, *, version=VERSION, compatibility=False):
    load_split(root, version=version)
    path = _version_dir(root, version) / ("manifest_legacy.csv" if compatibility else "manifest.csv")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows


def activate_split(root, *, version=VERSION):
    """Set only an explicit next-run pointer; running jobs and source CSV are untouched."""
    root = Path(root).resolve()
    report = load_split(root, version=version)
    pointer = {"schema_version": 1, "version": version, "manifest": report["manifest_relative_path"],
        "report": f"data/training/{version}/report.json", "status": "ready_for_next_training",
        "training_started": False, "independent_test": False, "diagnostic_only": True,
        "requires_diagnostic_test_mode": True, "source_manifest_sha256": report["source_manifest_sha256"]}
    target = root / "data/training/active_split.json"
    fd, name = tempfile.mkstemp(prefix=".active-split-", suffix=".json", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(_json(pointer))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, target)
    finally:
        if os.path.exists(name):
            os.unlink(name)
    return pointer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--version", default=VERSION)
    parser.add_argument("--seed", default=SEED)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--activate", action="store_true", help="Point next-run configuration to this version; does not launch training")
    args = parser.parse_args()
    report = load_split(args.root, version=args.version) if args.check else prepare(args.root, version=args.version, seed=args.seed)
    if args.activate:
        activate_split(args.root, version=args.version)
    print(json.dumps({k: report[k] for k in ("version", "counts", "independent_test", "diagnostic_only", "source_manifest_sha256")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
