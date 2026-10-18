"""Boundary tests: human authority, leakage, immutable exports and real decoding."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile

from export_human_dataset import (
    ACTION_TO_ID, DatasetExportError, HumanActionDataset, export_snapshot, validate_snapshot,
)
from human_review import HumanReview


class HumanDatasetExportTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.report = self.root / "data/autolabel/detailed_train_report"
        self.report.mkdir(parents=True)
        for video in ("train", "val"):
            path = self.root / f"data/raw/open/videos/{video}.mov"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"synthetic {video} content".encode())
        self.manifest = self.root / "data/training/manifest.csv"
        self.manifest.parent.mkdir(parents=True)
        self.write_manifest()
        for name, content in (("data/review/event_review.csv", "synthetic\n"),
                              ("data/transcripts/observations.jsonl", ""),
                              ("data/transcripts/sources.json", "{}")):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        self.train = {"video": "videos/train.mov", "group_id": "video_1111111111111111", "split": "train",
                      "start_sec": 2, "end_sec": 5}
        self.val = dict(self.train, video="videos/val.mov", group_id="video_2222222222222222", split="val")
        self.write("video_timelines.jsonl", [dict(row, duration_sec=10, preview=f"previews/{row['group_id']}.mp4",
            preview_metadata={"clip_first_source_sec": 0}) for row in [self.train, self.val]])
        self.write("user_reference_alignment.jsonl", [dict(row, reference_id="ref_" + row["split"],
            source_action="cash_received", source_status="transcribed", source_observation="Synthetic action.",
            model_support_status="not_recovered_by_model") for row in [self.train, self.val]])
        self.write("moments.jsonl", [dict(self.train, moment_id="moment_1", observed_action="cash_received",
            objects=["cash"], conflict_ids=[], evidence_timestamps=[3])])
        self.write("review_queue.jsonl", [dict(self.train, review_id="queue_1", conflict_ids=[], observations=[], reasons=[])])
        self.store = HumanReview(self.root)

    def write(self, name, rows):
        (self.report / name).write_text("".join(json.dumps(row) + "\n" for row in rows))

    def write_manifest(self):
        self.manifest.write_text("video,group_id,sha256,split\n" + "".join(
            f"videos/{name}.mov,video_{'1' if name == 'train' else '2'}{'1' * 15 if name == 'train' else '2' * 15},"
            f"{hashlib.sha256((self.root / f'data/raw/open/videos/{name}.mov').read_bytes()).hexdigest()},{name}\n"
            for name in ("train", "val")))

    def payload(self, source_kind="user_reference", **changes):
        task = next(t for t in self.store.catalog()["tasks"] if t["source_kind"] == source_kind)
        data = {"task_id": task["id"], "source_fingerprint": task["source_fingerprint"], "expected_version": 0,
                "reviewer": "Reviewer A", "decision": "accept", "start_sec": 2, "end_sec": 5,
                "observed_action": "cash_received", "objects": ["cash"], "note": "Synthetic reviewed action.",
                "evidence_time_sec": 3, "visibility": "clear", "event_assessment": "not_assessed", "violation_type": None}
        data.update(changes)
        return data

    def export(self, **kwargs):
        return export_snapshot(self.root, store=self.store, **kwargs)

    def rows(self, meta):
        return [json.loads(line) for line in (Path(meta["path"]) / "manifest.jsonl").read_text().splitlines()]

    def test_empty_snapshot_is_valid_but_cannot_train(self):
        meta = self.export()
        self.assertEqual(meta["count"], 0)
        self.assertEqual(meta["status"], "empty_waiting_for_human_review")
        self.assertFalse(meta["training_started"])
        self.assertTrue(validate_snapshot(meta["path"], self.root)["valid"])
        with self.assertRaisesRegex(DatasetExportError, "empty snapshot"):
            HumanActionDataset(self.root, meta["path"])

    def test_only_explicit_accept_is_exported_with_provenance(self):
        self.store.save(self.payload(event_assessment="normal_action"))
        self.store.save(self.payload(source_kind="model_moment", decision="reject"))
        self.store.save(self.payload(source_kind="compact_review", decision="needs_context"))
        meta = self.export()
        row, = self.rows(meta)
        self.assertEqual(meta["excluded_decisions"], {"needs_context": 1, "reject": 1})
        self.assertEqual(row["observed_action"], "cash_received")
        self.assertEqual(row["event_assessment"], "normal_action")
        self.assertEqual(row["label_scope"], "observed_action")
        self.assertFalse(row["normal_action_is_whole_video_label"])
        self.assertFalse(row["violation_confirmed"])
        self.assertNotIn("target", row)
        self.assertEqual(row["reviewer"], "Reviewer A")
        self.assertEqual(row["task_version"], 1)
        self.assertEqual(row["evidence_time_sec"], 3)
        self.assertEqual(len(row["provenance"]["decision_sha256"]), 64)
        self.assertTrue(validate_snapshot(meta["path"], self.root, require_nonempty=True)["valid"])

    def test_possible_violation_is_preserved_as_hypothesis_only(self):
        self.store.save(self.payload(event_assessment="possible_violation", violation_type="previous_cash_left"))
        row, = self.rows(self.export())
        self.assertEqual(row["possible_violation_type"], "previous_cash_left")
        self.assertFalse(row["eligible_for_violation_training"])
        self.assertFalse(row["violation_confirmed"])
        self.assertEqual(row["action_id"], ACTION_TO_ID["cash_received"])

    def test_exact_duplicate_annotations_merge_with_all_review_audit(self):
        self.store.save(self.payload())
        self.store.save(self.payload(source_kind="model_moment", reviewer="Reviewer B", evidence_time_sec=4))
        meta = self.export()
        row, = self.rows(meta)
        self.assertEqual(meta["accepted_review_decisions"], 2)
        self.assertEqual(meta["exact_duplicate_annotations_merged"], 1)
        self.assertEqual(len(row["source_task_ids"]), 2)
        self.assertEqual({r["reviewer"] for r in row["review_audit"]}, {"Reviewer A", "Reviewer B"})
        self.assertEqual({r["evidence_time_sec"] for r in row["review_audit"]}, {3, 4})
        self.assertTrue(validate_snapshot(meta["path"], self.root)["valid"])
        self.store.save(self.payload(source_kind="model_moment", expected_version=1, decision="reject"))
        with self.assertRaisesRegex(DatasetExportError, "retracted"):
            validate_snapshot(meta["path"], self.root)

    def test_partial_overlap_is_not_merged_or_assumed_negative(self):
        self.store.save(self.payload())
        self.store.save(self.payload(source_kind="model_moment", start_sec=4, end_sec=6, evidence_time_sec=5))
        meta = self.export()
        self.assertEqual(meta["count"], 2)
        self.assertEqual(meta["unique_source_videos"], 1)
        self.assertEqual(meta["overlap_checks"]["same_action_overlapping_pairs"], 1)
        self.assertEqual(sum(r["group_balanced_weight"] for r in self.rows(meta)), 1)
        self.assertTrue(validate_snapshot(meta["path"], self.root)["valid"])

    def test_source_and_context_changes_invalidate_snapshot(self):
        self.store.save(self.payload())
        meta = self.export()
        path = self.root / "data/transcripts/observations.jsonl"
        path.write_text("\n")
        with self.assertRaisesRegex(DatasetExportError, "Protected"):
            validate_snapshot(meta["path"], self.root)
        path.write_text("")
        with (self.report / "moments.jsonl").open("a") as handle:
            handle.write("\n")
        with self.assertRaisesRegex(DatasetExportError, "context changed"):
            validate_snapshot(meta["path"], self.root)
        new = self.export()
        self.assertEqual(new["count"], 0)
        self.assertEqual(new["excluded_decisions"], {"stale_or_removed_task": 1})

    def test_edited_video_rejected_even_when_manifest_is_unchanged(self):
        self.store.save(self.payload())
        meta = self.export()
        (self.root / "data/raw/open/videos/train.mov").write_bytes(b"changed content")
        with self.assertRaisesRegex(DatasetExportError, "Source video"):
            validate_snapshot(meta["path"], self.root)
        with self.assertRaisesRegex(DatasetExportError, "Source video"):
            self.export()

    def test_new_unrelated_review_does_not_invalidate_but_edit_does(self):
        self.store.save(self.payload())
        meta = self.export()
        self.store.save(self.payload(source_kind="model_moment", decision="reject"))
        self.assertTrue(validate_snapshot(meta["path"], self.root)["valid"])
        self.store.save(self.payload(expected_version=1, note="Boundary corrected.", start_sec=2.5))
        with self.assertRaisesRegex(DatasetExportError, "review changed"):
            validate_snapshot(meta["path"], self.root)

    def test_archive_has_only_metadata_and_new_snapshot_never_overwrites(self):
        self.store.save(self.payload())
        before = {p: p.read_bytes() for p in self.report.iterdir()}
        meta = self.export(snapshot_id="safe_snapshot")
        original = (Path(meta["path"]) / "metadata.json").read_bytes()
        with zipfile.ZipFile(self.root / meta["archive_relative_path"]) as archive:
            self.assertEqual(set(archive.namelist()), {"manifest.jsonl", "metadata.json", "label_map.json"})
        with self.assertRaises(FileExistsError):
            self.export(snapshot_id="safe_snapshot")
        self.assertEqual(original, (Path(meta["path"]) / "metadata.json").read_bytes())
        self.assertEqual(before, {p: p.read_bytes() for p in self.report.iterdir()})

    def test_split_leakage_and_corrupt_approved_bounds_fail_closed(self):
        self.store.save(self.payload())
        catalog = self.store.catalog()
        corrupt = copy.deepcopy(catalog)
        next(iter(corrupt["decisions"].values()))["end_sec"] = 11
        with mock.patch.object(self.store, "catalog", return_value=corrupt):
            with self.assertRaisesRegex(DatasetExportError, "bounds"):
                self.export()
        content = self.manifest.read_text()
        train_sha = content.splitlines()[1].split(",")[2]
        val_sha = content.splitlines()[2].split(",")[2]
        self.manifest.write_text(content.replace(val_sha, train_sha))
        with self.assertRaisesRegex(DatasetExportError, "leaks"):
            self.export()

    def test_validation_video_cannot_be_injected_as_action_training(self):
        self.store.save(self.payload())
        catalog = self.store.catalog()
        decision = next(iter(catalog["decisions"].values()))
        task = next(t for t in catalog["tasks"] if t["id"] == decision["task_id"])
        for target in (decision, task):
            target.update(video="videos/val.mov", group_id=self.val["group_id"], split="val")
        with mock.patch.object(self.store, "catalog", return_value=catalog):
            with self.assertRaisesRegex(DatasetExportError, "frozen train"):
                self.export()

    def test_concurrent_review_change_aborts_without_partial_snapshot(self):
        self.store.save(self.payload())
        first = self.store.catalog()
        changed = copy.deepcopy(first)
        next(iter(changed["decisions"].values()))["version"] += 1
        with mock.patch.object(self.store, "catalog", side_effect=[first, changed]):
            with self.assertRaisesRegex(DatasetExportError, "during export"):
                self.export(snapshot_id="raced")
        exports = self.root / "data/human_review/exports"
        self.assertEqual([p.name for p in exports.iterdir()], [".export.lock"])

    def test_artifact_edit_detected_and_class_ids_do_not_depend_on_samples(self):
        empty = self.export()
        self.store.save(self.payload())
        nonempty = self.export()
        self.assertEqual((Path(empty["path"]) / "label_map.json").read_bytes(),
                         (Path(nonempty["path"]) / "label_map.json").read_bytes())
        path = Path(nonempty["path"]) / "manifest.jsonl"
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(DatasetExportError, "artifact content"):
            validate_snapshot(nonempty["path"], self.root)

    def test_lazy_loader_decodes_context_and_preserves_aspect_ratio(self):
        try:
            import av
            import numpy as np
            import torch
        except ImportError:
            self.skipTest("Optional decoder/torch dependencies unavailable")
        source = self.root / "data/raw/open/videos/train.mov"
        with av.open(str(source), "w") as container:
            stream = container.add_stream("mpeg4", rate=10)
            stream.width, stream.height, stream.pix_fmt = 64, 32, "yuv420p"
            for index in range(100):
                array = np.zeros((32, 64, 3), dtype=np.uint8)
                array[:, :, 0] = 180
                array[:, :, 1] = index
                frame = av.VideoFrame.from_ndarray(array, format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        self.write_manifest()
        self.store.save(self.payload(event_assessment="normal_action"))
        meta = self.export()
        dataset = HumanActionDataset(self.root, meta["path"], num_frames=8, context_seconds=1)
        sample = dataset[0]
        self.assertEqual(tuple(sample["pixel_values"].shape), (3, 8, 384, 384))
        self.assertEqual(sample["action_id"].item(), ACTION_TO_ID["cash_received"])
        self.assertEqual(sample["event_assessment"], "normal_action")
        self.assertTrue(torch.allclose(sample["pixel_values"][:, :, :90, :], torch.full((3, 8, 90, 384), 114 / 255)))
        self.assertTrue(bool(sample["focus_mask"].any()))
        self.assertTrue(bool((~sample["focus_mask"]).any()))
        self.assertGreaterEqual(sample["frame_timestamps"].min().item(), 1)
        self.assertLessEqual(sample["frame_timestamps"].max().item(), 6)
        batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=1)))
        self.assertEqual(tuple(batch["pixel_values"].shape), (1, 3, 8, 384, 384))
        self.assertNotIn("violation_target", sample)
        full_context = HumanActionDataset(self.root, meta["path"], num_frames=8, context_seconds=6)[0]
        self.assertEqual(full_context["frame_timestamps"][-1].item(), 9.9)
        self.assertEqual(full_context["context_bounds"][-1].item(), 10)


if __name__ == "__main__":
    unittest.main()
