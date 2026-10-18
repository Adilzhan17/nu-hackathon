import csv
import tempfile
import unittest
from pathlib import Path

from check_dataset import validate


class DatasetValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ("a.mov", "b.mov", "c.mov", "d.mov"):
            (self.root / name).write_bytes(name.encode())
        self.rows = [
            ["a.mov", "take1", "train", 0, 2, "normal"],
            ["b.mov", "take2", "train", 0, 2, "event"],
            ["c.mov", "take3", "val", 0, 2, "normal"],
            ["d.mov", "take4", "val", 0, 2, "event"],
        ]

    def report(self):
        manifest = self.root / "annotations.csv"
        with manifest.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["video", "group_id", "split", "start_sec", "end_sec", "label"])
            writer.writerows(self.rows)
        return validate(manifest, self.root)

    def test_separate_groups(self):
        self.assertTrue(self.report()["valid"])

    def test_same_take_cannot_cross_splits(self):
        self.rows[2][1] = "take1"
        self.assertFalse(self.report()["valid"])

    def test_renamed_duplicate_cannot_cross_splits(self):
        (self.root / "c.mov").write_bytes((self.root / "a.mov").read_bytes())
        self.assertTrue(any("leaks" in e for e in self.report()["errors"]))

    def test_nonfinite_time_rejected(self):
        self.rows[0][4] = "nan"
        self.assertFalse(self.report()["valid"])

    def test_conflicting_intervals_rejected(self):
        self.rows.append(["a.mov", "take1", "train", 1, 3, "event"])
        self.assertTrue(any("overlapping" in e for e in self.report()["errors"]))

    def test_relative_path_escape_rejected(self):
        self.rows[0][0] = "../outside.mov"
        self.assertFalse(self.report()["valid"])

    def test_adjacent_intervals_allowed(self):
        self.rows.append(["a.mov", "take1", "train", 2, 4, "event"])
        self.assertTrue(self.report()["valid"])

    def test_unseen_validation_type_rejected(self):
        self.rows[3][5] = "unseen"
        self.assertTrue(any("absent" in e for e in self.report()["errors"]))


if __name__ == "__main__":
    unittest.main()
