"""Training-boundary checks; these tests never acquire the real GPU lock."""
import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from .train import assert_gpu_idle, canonical_sha, gpu_lock, single_example_collate, validate_rows


class TrainingSafetyTests(unittest.TestCase):
    def rows_protocol(self):
        protocol = {"group_membership": {"g1": "fit", "g2": "inner_holdout"},
                    "source_identity": {"g1": {"split": "train", "video_sha256": "source-one"},
                                        "g2": {"split": "train", "video_sha256": "source-two"}}}
        protocol["protocol_sha256"] = canonical_sha(protocol)
        rows = [{"sample_id": "sample-one", "group_id": "g1", "video_sha256": "source-one",
                 "partition": "fit", "kind": "human_action", "trust_level": "human_confirmed",
                 "loss_weight": 1.0, "prompt": "ordered frames", "target": "visible action"}]
        return rows, protocol

    def test_holdout_and_identity_changes_fail_closed(self):
        rows, protocol = self.rows_protocol()
        self.assertEqual(validate_rows(rows, protocol)["eligible_examples"], 1)
        for changes in ({"group_id": "g2"}, {"video_sha256": "different-source"}, {"partition": "inner_holdout"}):
            altered = [dict(rows[0], **changes)]
            with self.assertRaises(ValueError):
                validate_rows(altered, protocol)

    def test_weak_sources_cannot_increase_trust_or_weight(self):
        rows, protocol = self.rows_protocol()
        rows[0].update(kind="weak_caption", trust_level="weak_unconfirmed", loss_weight=0.2)
        validate_rows(rows, protocol)
        for changes in ({"loss_weight": 1.0}, {"trust_level": "human_confirmed"}, {"status": "quarantined"}):
            with self.assertRaises(ValueError):
                validate_rows([dict(rows[0], **changes)], protocol)

    def test_modified_partition_and_duplicate_samples_are_rejected(self):
        rows, protocol = self.rows_protocol()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_rows(rows + copy.deepcopy(rows), protocol)
        protocol["group_membership"]["g2"] = "fit"
        with self.assertRaisesRegex(ValueError, "digest"):
            validate_rows(rows, protocol)

    def test_lock_busy_and_exception_release(self):
        with tempfile.TemporaryDirectory() as temporary:
            lock = Path(temporary) / "gpu.lock"
            with gpu_lock(lock):
                with self.assertRaisesRegex(RuntimeError, "GPU_BUSY"):
                    with gpu_lock(lock):
                        self.fail("Concurrent job obtained common GPU lock")
            with self.assertRaisesRegex(ValueError, "intentional"):
                with gpu_lock(lock):
                    raise ValueError("intentional")
            with gpu_lock(lock):
                pass

    def test_foreign_gpu_process_is_not_terminated(self):
        result = subprocess.CompletedProcess([], 0, stdout="12345, other-training, 10000\n", stderr="")
        with patch("subprocess.run", return_value=result) as run:
            with self.assertRaisesRegex(RuntimeError, "GPU_BUSY"):
                assert_gpu_idle()
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args.args[0][0], "nvidia-smi")

    def test_collator_preserves_frame_lists_and_requires_batch_one(self):
        sample = {"frames": [object(), object()], "prompt": "ordered frames"}
        self.assertIs(single_example_collate([sample]), sample)
        with self.assertRaises(ValueError):
            single_example_collate([sample, sample])


if __name__ == "__main__":
    unittest.main()
