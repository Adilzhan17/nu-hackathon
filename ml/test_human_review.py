import concurrent.futures
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from human_review import HumanReview, ReviewError


class HumanReviewTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.report = self.root / "data/autolabel/detailed_train_report"
        self.report.mkdir(parents=True)
        manifest = self.root / "data/training/manifest.csv"
        manifest.parent.mkdir(parents=True)
        manifest.write_text("video,group_id,sha256,split\nvideos/train.mov,video_1111111111111111," + "a" * 64 + ",train\nvideos/val.mov,video_2222222222222222," + "b" * 64 + ",val\n")
        self.train = {"video": "videos/train.mov", "group_id": "video_1111111111111111", "split": "train", "start_sec": 2, "end_sec": 5}
        self.val = {"video": "videos/val.mov", "group_id": "video_2222222222222222", "split": "val", "start_sec": 2, "end_sec": 5}
        self.write("video_timelines.jsonl", [dict(r, duration_sec=10,
            preview=f"previews/{r['group_id']}.mp4", preview_metadata={"clip_first_source_sec": 0}) for r in [self.train, self.val]])
        self.write("user_reference_alignment.jsonl", [dict(r, reference_id="ref_" + r["split"],
            source_action="cash_received", source_status="transcribed", source_observation="Synthetic action.",
            model_support_status="not_recovered_by_model") for r in [self.train, self.val]])
        self.write("moments.jsonl", [dict(self.train, moment_id="moment1", observed_action="document_transfer", objects=["document"], conflict_ids=["c1"], evidence_timestamps=[3])])
        self.write("review_queue.jsonl", [dict(self.train, review_id="q1", conflict_ids=["c1"], observations=[], reasons=["possible_object_disagreement"]),
            dict(self.train, review_id="quarantined", observations=[{"status": "quarantined"}], reasons=[])])
        self.store = HumanReview(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, rows):
        (self.report / name).write_text("".join(json.dumps(r) + "\n" for r in rows))

    def task(self):
        return next(t for t in self.store.catalog()["tasks"] if t["source_kind"] == "user_reference")

    def payload(self, **changes):
        task = self.task()
        value = {"task_id": task["id"], "source_fingerprint": task["source_fingerprint"], "expected_version": 0,
            "reviewer": "AB", "decision": "accept", "start_sec": 2, "end_sec": 5,
            "observed_action": "cash_received", "objects": ["cash"], "note": "", "evidence_time_sec": 3,
            "visibility": "clear", "event_assessment": "not_assessed", "violation_type": None}
        value.update(changes)
        return value

    def test_train_only_priority_and_source_integrity(self):
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.report.iterdir()}
        catalog = self.store.catalog()
        self.assertEqual(len(catalog["tasks"]), 3)
        self.assertEqual(catalog["tasks"][0]["source_kind"], "compact_review")
        self.assertTrue(all(t["split"] == "train" for t in catalog["tasks"]))
        self.assertTrue(all(t["source_id"] != "quarantined" for t in catalog["tasks"]))
        self.store.save(self.payload())
        self.assertEqual(before, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.report.iterdir()})

    def test_recommended_queue_avoids_overlapping_duplicate_proposals(self):
        catalog = self.store.catalog()
        self.assertEqual(len(catalog["recommended_task_ids"]), 1)
        self.assertEqual(catalog["recommended_task_ids"][0], catalog["tasks"][0]["id"])
        self.assertEqual(len(catalog["tasks"]), 3)

    def test_confirmed_episode_alternatives_remain_pending_but_leave_recommendations(self):
        self.store.save(self.payload())
        catalog = self.store.catalog()
        self.assertEqual(catalog["recommended_task_ids"], [])
        self.assertEqual(catalog["counts"]["pending"], 2)
        self.assertEqual(catalog["counts"]["confirmed_actions"], 1)
        self.assertEqual(len(catalog["decisions"]), 1)

    def test_accept_reloads_from_journal_and_retains_created_time(self):
        saved = self.store.save(self.payload())["decision"]
        self.assertTrue(saved["action_confirmed"])
        self.assertFalse(saved["violation_confirmed"])
        self.assertEqual(saved["video_sha256"], "a" * 64)
        self.store.snapshot.unlink()
        second = HumanReview(self.root)
        self.assertEqual(second.catalog()["decisions"][saved["task_id"]], saved)
        updated = second.save(self.payload(expected_version=1, note="Boundary checked."))["decision"]
        self.assertEqual(updated["created_at"], saved["created_at"])
        self.assertEqual(updated["version"], 2)
        self.assertEqual(len(self.store.journal.read_text().splitlines()), 2)

    def test_stale_version_conflict(self):
        self.store.save(self.payload())
        with self.assertRaises(ReviewError) as caught:
            self.store.save(self.payload())
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.code, "stale_version")

    def test_source_change_invalidates_old_page_and_export(self):
        payload = self.payload()
        self.store.save(payload)
        with (self.report / "moments.jsonl").open("a") as handle:
            handle.write("\n")
        with self.assertRaises(ReviewError) as caught:
            self.store.save(dict(payload, expected_version=1))
        self.assertEqual(caught.exception.code, "source_changed")
        catalog = self.store.catalog()
        self.assertEqual(catalog["counts"]["stale"], 1)
        self.assertEqual(catalog["counts"]["confirmed_actions"], 0)
        exported = json.loads(self.store.export_jsonl())
        self.assertTrue(exported["source_stale"])
        self.assertFalse(exported["eligible_for_action_training"])

    def test_invalid_accept_does_not_persist(self):
        invalid = [dict(visibility="unclear"), dict(observed_action="unclear_action"),
            dict(evidence_time_sec=None), dict(evidence_time_sec=6), dict(objects=["document"]),
            dict(observed_action="other_observed_action", objects=[]), dict(end_sec=11),
            dict(start_sec=float("nan")), dict(evidence_time_sec=True), dict(reviewer=" "),
            dict(expected_version=True), dict(observed_action="invented_action")]
        for changes in invalid:
            with self.subTest(changes=changes):
                with self.assertRaises(ReviewError):
                    self.store.save(self.payload(**changes))
        self.assertEqual(self.store.catalog()["counts"]["reviewed"], 0)

    def test_partial_visibility_can_confirm_action(self):
        result = self.store.save(self.payload(visibility="partial"))
        self.assertEqual(result["counts"]["confirmed_actions"], 1)

    def test_uncertainty_and_reject_are_not_negative_ground_truth(self):
        for idx, decision in enumerate(["needs_context", "reject"]):
            saved = self.store.save(self.payload(expected_version=idx, decision=decision,
                visibility="unclear" if decision == "needs_context" else "clear",
                observed_action=None, evidence_time_sec=None, objects=[], event_assessment="normal_action"))["decision"]
            self.assertFalse(saved["action_confirmed"])
            self.assertFalse(saved["eligible_for_action_training"])
            self.assertFalse(saved["eligible_for_violation_training"])
            self.assertFalse(saved["rejected_proposal_is_negative_label"])
            self.assertEqual(saved["event_assessment"], "not_assessed")

    def test_normal_action_does_not_certify_whole_video(self):
        saved = self.store.save(self.payload(event_assessment="normal_action"))["decision"]
        self.assertFalse(saved["normal_action_is_whole_video_label"])
        self.assertFalse(saved["eligible_for_violation_training"])

    def test_possible_violation_is_human_hypothesis_with_fixed_type(self):
        with self.assertRaises(ReviewError):
            self.store.save(self.payload(event_assessment="possible_violation", violation_type="unregistered_type"))
        saved = self.store.save(self.payload(event_assessment="possible_violation", violation_type="previous_cash_left"))["decision"]
        self.assertEqual(saved["assessment_authority"], "human_review")
        self.assertFalse(saved["violation_confirmed"])

    def test_manual_action_persists_and_requires_separate_confirmation(self):
        base = self.task()
        result = self.store.add_task({"base_task_id": base["id"], "source_fingerprint": base["source_fingerprint"],
            "reviewer": "AB", "start_sec": 6, "end_sec": 8, "description": "Missed event."})
        task = result["task"]
        self.assertEqual(task["source_kind"], "human_added")
        self.assertEqual(task["suggested_action"], "unclear_action")
        catalog = HumanReview(self.root).catalog()
        self.assertTrue(any(t["id"] == task["id"] for t in catalog["tasks"]))
        self.assertNotIn(task["id"], catalog["decisions"])
        self.assertEqual(catalog["counts"]["confirmed_actions"], 0)
        saved = self.store.save(self.payload(task_id=task["id"], start_sec=6, end_sec=8, evidence_time_sec=7))["decision"]
        self.assertEqual(saved["video"], base["video"])

    def test_manual_task_cannot_inject_validation_or_arbitrary_path(self):
        with self.assertRaises(ReviewError):
            self.store.add_task({"base_task_id": "unknown", "source_fingerprint": "x", "reviewer": "AB", "video": "/etc/passwd", "start_sec": 0, "end_sec": 2})
        base = self.task()
        with self.assertRaises(ReviewError):
            self.store.add_task({"base_task_id": base["id"], "source_fingerprint": base["source_fingerprint"], "reviewer": "AB", "start_sec": 0, "end_sec": 20})

    def test_concurrent_edit_only_one_wins(self):
        payload = self.payload()
        def submit(_):
            try:
                return HumanReview(self.root).save(payload)["decision"]["version"]
            except ReviewError as exc:
                return exc.code
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(submit, range(2)))
        self.assertCountEqual(results, [1, "stale_version"])
        self.assertEqual(len(self.store.journal.read_text().splitlines()), 1)

    def test_different_tasks_keep_independent_versions_and_both_changes(self):
        tasks = self.store.catalog()["tasks"][:2]
        payloads = [self.payload(task_id=t["id"], observed_action="document_transfer", objects=["document"]) for t in tasks]
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda p: HumanReview(self.root).save(p), payloads))
        self.assertEqual([r["decision"]["version"] for r in results], [1, 1])
        self.assertEqual(sorted(r["revision"] for r in results), [1, 2])
        self.assertEqual(self.store.catalog()["counts"]["confirmed_actions"], 2)

    def test_successful_journal_survives_snapshot_failure(self):
        with mock.patch.object(self.store, "_write_snapshot", side_effect=OSError("disk error")):
            with self.assertRaises(OSError):
                self.store.save(self.payload())
        self.assertEqual(HumanReview(self.root).catalog()["counts"]["confirmed_actions"], 1)

    def test_export_latest_decision_only(self):
        self.store.save(self.payload())
        self.store.save(self.payload(expected_version=1, decision="reject", note="Wrong proposal."))
        rows = [json.loads(s) for s in self.store.export_jsonl().splitlines()]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["version"], 2)
        self.assertFalse(rows[0]["eligible_for_action_training"])


if __name__ == "__main__":
    unittest.main()
