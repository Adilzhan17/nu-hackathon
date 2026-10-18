import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import random
import tempfile
import unittest

from split95 import (SOURCE, SplitError, activate_split, apply_split, build_report,
                     filter_human_examples, load_manifest, load_split, prepare)
from train_suite import fit_fixed_diagnostic_head, read_training_manifest, resolve_training_config


class Split95Tests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / SOURCE
        self.source.parent.mkdir(parents=True)
        self.rows = []
        for index in range(40):
            # Unequal window counts deliberately make a row-level 95/5 split wrong.
            for window in range(1 if index % 3 else 4):
                self.rows.append(dict(video=f"videos/synthetic_{index}.mov", group_id=f"group_{index}",
                    sha256=hashlib.sha256(f"synthetic-{index}".encode()).hexdigest(),
                    split="train" if index < 32 else "val", start_sec=str(window * 2), end_sec=str(window * 2 + 2),
                    target=str(index % 2), label="normal" if index % 2 == 0 else "violation_unspecified"))
        self.write(self.rows)

    def write(self, rows):
        with self.source.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def report(self, rows=None):
        return build_report(self.rows if rows is None else rows, hashlib.sha256(self.source.read_bytes()).hexdigest())

    def test_38_2_by_source_video_and_balanced_old_val_diagnostic(self):
        report = self.report()
        self.assertEqual(report["counts"]["train"]["unique_videos"], 38)
        self.assertEqual(report["counts"]["diagnostic_test"]["unique_videos"], 2)
        heldout = [e for e in report["entities"] if e["split"] == "diagnostic_test"]
        self.assertEqual({e["source_split"] for e in heldout}, {"val"})
        self.assertEqual({e["source_target"] for e in heldout}, {0, 1})
        self.assertFalse(report["independent_test"])
        self.assertFalse(report["selection_uses_model_metrics"])
        self.assertEqual(len(report["promotion_from_old_validation"]), 6)
        self.assertEqual(report["total_windows"], len(self.rows))

    def test_selection_deterministic_under_row_order_and_window_duplication(self):
        shuffled = list(self.rows)
        random.Random(99).shuffle(shuffled)
        shuffled.extend(copy.deepcopy(self.rows[:4]))
        self.assertEqual(self.report()["sha256_membership"], self.report(shuffled)["sha256_membership"])

    def test_aliases_and_groups_with_identical_sha_are_indivisible(self):
        alias = dict(self.rows[-1], video="videos/alias.mov", group_id="alias_group")
        report = self.report(self.rows + [alias])
        self.assertEqual(report["total_unique_videos"], 40)
        self.assertEqual(report["total_source_groups"], 41)
        self.assertEqual(report["group_membership"][alias["group_id"]], report["group_membership"][self.rows[-1]["group_id"]])
        converted = apply_split(self.rows + [alias], report)
        self.assertEqual(len({r["split"] for r in converted if r["sha256"] == alias["sha256"]}), 1)

    def test_source_leakage_conflicting_labels_and_identity_fail_closed(self):
        invalid = [dict(self.rows[0], split="val", group_id="leaked_alias"),
                   dict(self.rows[0], target="1"), dict(self.rows[0], sha256="f" * 64),
                   dict(self.rows[0], video="../outside.mov")]
        for bad in invalid:
            with self.subTest(bad=bad):
                with self.assertRaises(SplitError):
                    self.report(self.rows + [bad])

    def test_prepare_keeps_original_and_immutable_artifacts(self):
        original = self.source.read_bytes()
        report = prepare(self.root)
        self.assertEqual(self.source.read_bytes(), original)
        self.assertEqual(load_split(self.root), report)
        self.assertEqual(prepare(self.root), report)
        canonical = load_manifest(self.root)
        legacy = load_manifest(self.root, compatibility=True)
        self.assertEqual({r["split"] for r in canonical}, {"train", "diagnostic_test"})
        self.assertEqual({r["split"] for r in legacy}, {"train", "val"})
        for before, after in zip(self.rows, canonical):
            self.assertEqual({k: v for k, v in before.items() if k != "split"}, {k: v for k, v in after.items() if k != "split"})
        with self.assertRaises(SplitError):
            prepare(self.root, seed="different")

    def test_changed_source_or_modified_artifact_is_detected(self):
        prepare(self.root)
        original = self.source.read_bytes()
        self.source.write_bytes(original + b"\n")
        with self.assertRaises(SplitError):
            load_split(self.root)
        self.source.write_bytes(original)
        path = self.root / "data/training/split95_v1/manifest.csv"
        path.write_text(path.read_text() + "\n")
        with self.assertRaises(SplitError):
            load_split(self.root)

    def test_active_pointer_does_not_launch_or_overwrite_original(self):
        original = self.source.read_bytes()
        prepare(self.root)
        pointer = activate_split(self.root)
        self.assertFalse(pointer["training_started"])
        self.assertTrue(pointer["requires_diagnostic_test_mode"])
        self.assertEqual(self.source.read_bytes(), original)
        self.assertEqual(json.loads((self.root / "data/training/active_split.json").read_text()), pointer)

    def test_training_defaults_consume_active_policy_but_explicit_source_keeps_legacy(self):
        path, diagnostic, policy = resolve_training_config(self.root)
        self.assertEqual(path, self.source)
        self.assertFalse(diagnostic)
        prepare(self.root)
        activate_split(self.root)
        path, diagnostic, policy = resolve_training_config(self.root)
        self.assertEqual(path, self.root / "data/training/split95_v1/manifest.csv")
        self.assertTrue(diagnostic)
        self.assertTrue(policy["active_policy_used"])
        path, diagnostic, policy = resolve_training_config(self.root, manifest=self.source)
        self.assertEqual(path, self.source)
        self.assertFalse(diagnostic)
        self.assertFalse(policy["active_policy_used"])

    def test_active_policy_tampering_fails_closed(self):
        prepare(self.root)
        pointer = activate_split(self.root)
        active = self.root / "data/training/active_split.json"
        for change in [dict(manifest="../../outside.csv"), dict(source_manifest_sha256="0" * 64),
                       dict(requires_diagnostic_test_mode=False), dict(independent_test=True)]:
            active.write_text(json.dumps(dict(pointer, **change)))
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    resolve_training_config(self.root)

    def test_human_export_filter_excludes_heldout_and_preserves_labels(self):
        report = self.report()
        records = [dict(video=e["videos"][0], group_id=e["group_ids"][0], video_sha256=e["sha256"],
            split=e["source_split"], observed_action="cash_received", action_confirmed=True) for e in report["entities"]]
        selected, audit = filter_human_examples(records, report)
        self.assertEqual(len(selected), 38)
        self.assertEqual(audit["excluded_partitions"], {"diagnostic_test": 2})
        self.assertTrue(all(r["observed_action"] == "cash_received" for r in selected))
        self.assertTrue(all(r["split"] == "train" for r in selected))
        self.assertTrue(all("split_version" not in r for r in records))
        with self.assertRaises(SplitError):
            filter_human_examples([dict(records[0], video_sha256="0" * 64)], report)

    def test_exact_population_and_old_val_label_balance_required(self):
        with self.assertRaises(SplitError):
            self.report([r for r in self.rows if r["group_id"] != "group_39"])
        missing_class = [dict(r, target="0") if r["split"] == "val" else r for r in self.rows]
        with self.assertRaises(SplitError):
            self.report(missing_class)

    def test_training_loader_requires_diagnostic_mode_for_new_manifest(self):
        prepare(self.root)
        directory = self.root / "data/training/split95_v1"
        for name in ("manifest.csv", "manifest_legacy.csv"):
            with self.assertRaises(ValueError):
                read_training_manifest(directory / name)
            rows, part = read_training_manifest(directory / name, diagnostic_test=True)
            self.assertIn(part, {"diagnostic_test", "val"})
            self.assertEqual(len({r["sha256"] for r in rows if r["split"] == "train"}), 38)

    def test_fixed_epoch_optimizer_has_no_diagnostic_inputs_or_early_selection(self):
        import torch
        torch.manual_seed(7)
        head = torch.nn.Linear(3, 1)
        before = head.weight.detach().clone()
        features = torch.tensor([[1., 0., 0.], [0., 1., 0.], [.5, 0., 1.], [0., .5, 1.]])
        labels = torch.tensor([0., 1., 0., 1.])
        events = []
        history = fit_fixed_diagnostic_head(head, features, labels, epochs=3, emit=lambda **event: events.append(event))
        self.assertEqual(len(history), 3)
        self.assertEqual([r["epoch"] for r in history], [1, 2, 3])
        self.assertTrue(all("val_loss" not in r and "diagnostic_loss" not in r for r in history))
        self.assertFalse(torch.equal(before, head.weight))
        self.assertEqual(len(events), 3)


if __name__ == "__main__":
    unittest.main()
