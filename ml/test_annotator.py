import csv
import http.client
import tempfile
import threading
import unittest
from pathlib import Path
from http.server import ThreadingHTTPServer

from annotator import ReviewStore, make_handler, preview_id, suggested_window


class AnnotationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.path = self.root / "event_review.csv"
        row = dict(video="videos/a.mov", group_id="take1", split="", start_sec="", end_sec="",
                   label="", observation="", rule_id="", duration_sec="10", review_status="pending",
                   anchor_sec="9", source_verdict="нарушение")
        with self.path.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        self.store = ReviewStore(self.path)

    def payload(self, **changes):
        result = dict(index=0, revision=self.store.read()[2], review_status="reviewed",
                      start_sec=2, end_sec=5, label="normal", observation="Наблюдаемое действие")
        result.update(changes)
        return result

    def test_save_persists_and_backs_up(self):
        original = self.path.read_bytes()
        self.store.update(self.payload())
        self.assertEqual(self.store.read()[0][0]["review_status"], "reviewed")
        self.assertEqual(next((self.root / "backups").iterdir()).read_bytes(), original)

    def test_stale_editor_does_not_overwrite(self):
        payload = self.payload()
        self.store.update(payload)
        with self.assertRaises(FileExistsError):
            self.store.update(payload)

    def test_bad_interval_does_not_modify_csv(self):
        original = self.path.read_bytes()
        for end in (11, -1, float("nan")):
            with self.assertRaises(ValueError):
                self.store.update(self.payload(end_sec=end))
            self.assertEqual(self.path.read_bytes(), original)

    def test_save_and_another_event_keeps_reviewed_event(self):
        self.assertEqual(self.store.update(self.payload(action="save_and_duplicate")), 1)
        rows = self.store.read()[0]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["review_status"], "reviewed")
        self.assertEqual(rows[1]["review_status"], "pending")
        self.assertEqual(rows[0]["group_id"], rows[1]["group_id"])
        self.assertEqual(rows[1]["start_sec"], "")

    def test_related_videos_must_not_share_preview(self):
        self.assertNotEqual(preview_id({"video":"a.mov", "group_id":"scene1"}),
                            preview_id({"video":"b.mov", "group_id":"scene1"}))

    def test_quick_confirmation_requires_no_typing(self):
        self.store.update(self.payload(mode="quick", label="", observation="", start_sec="", end_sec=""))
        row = self.store.read()[0][0]
        self.assertEqual(row["label"], "violation_unspecified")
        self.assertEqual(row["observation"], "")
        self.assertEqual(row["annotation_mode"], "point_window")
        self.assertEqual(row["label_scope"], "binary")
        self.assertEqual(float(row["start_sec"]), 5)
        self.assertEqual(float(row["end_sec"]), 10)

    def test_quick_type_is_preserved(self):
        self.store.update(self.payload(mode="quick", label="Наблюдаемое действие", observation=""))
        row = self.store.read()[0][0]
        self.assertEqual(row["label_scope"], "event_type")
        self.assertEqual(row["label"], "Наблюдаемое действие")

    def test_source_normals_are_not_claimed_human_reviewed(self):
        rows, fields, _ = self.store.read()
        rows += [dict(rows[0], video="normal.mov", source_verdict="без нарушения", anchor_sec=""),
                 dict(rows[0], video="reviewed.mov", source_verdict="без нарушения", review_status="reviewed", label="manual")]
        with self.path.open("w", encoding="utf-8-sig", newline="") as out:
            writer = csv.DictWriter(out, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        self.store.update(self.payload(action="accept_source_normals"))
        result = self.store.read()[0]
        self.assertEqual(result[0]["review_status"], "pending")
        self.assertEqual(result[1]["review_status"], "source_labeled")
        self.assertEqual(result[1]["annotation_mode"], "source_video_label")
        self.assertEqual(result[2]["label"], "manual")

    def test_point_window_at_start(self):
        self.assertEqual(suggested_window({"anchor_sec":"0", "duration_sec":"20"}), (0,4))

    def test_video_range_and_origin_protection(self):
        video = self.root / "video_1234567890abcdef.mp4"
        video.write_bytes(b"0123456789")
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.store, self.root))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            connection = http.client.HTTPConnection(*server.server_address)
            connection.request("GET", "/video/" + video.name, headers={"Range": "bytes=2-5"})
            response = connection.getresponse()
            self.assertEqual(response.status, 206)
            self.assertEqual(response.read(), b"2345")
            self.assertEqual(response.getheader("Content-Range"), "bytes 2-5/10")
            connection.request("POST", "/api/save", body="{}", headers={"Origin":"https://unrelated.example"})
            response = connection.getresponse()
            self.assertEqual(response.status, 403)
            response.read()
            connection.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
