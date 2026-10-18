"""HTTP integration of human review, concurrency, and immutable downloads.

All videos, descriptions, frames and persisted decisions are synthetic fixtures.
"""
import hashlib
import http.client
from http.server import ThreadingHTTPServer
import io
import json
from pathlib import Path
import socket
import tempfile
import threading
import unittest
import zipfile

from PIL import Image

from annotator import ReviewStore, make_handler
from autolabel.dataset import snapshot
from human_review import HumanReview
from object_review import ObjectReview


class HumanRoutesTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.report = self.root / "data/autolabel/detailed_train_report"
        self.report.mkdir(parents=True)
        source = self.root / "data/raw/open/videos/train.mov"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"synthetic video bytes, not decoded by export")
        self.video_sha = hashlib.sha256(source.read_bytes()).hexdigest()
        manifest = self.root / "data/training/manifest.csv"
        manifest.parent.mkdir(parents=True)
        manifest.write_text("video,group_id,sha256,split\nvideos/train.mov,video_1111111111111111," + self.video_sha + ",train\n")
        for name, content in (("data/review/event_review.csv", "video,group_id,duration_sec\nvideos/train.mov,video_1111111111111111,10\n"),
                              ("data/transcripts/observations.jsonl", ""),
                              ("data/transcripts/sources.json", "{}")):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        base = dict(video="videos/train.mov", group_id="video_1111111111111111", split="train", start_sec=2, end_sec=5)
        self.write("video_timelines.jsonl", [dict(base, duration_sec=10, preview="previews/video_1111111111111111.mp4", preview_metadata={"clip_first_source_sec": 0})])
        self.write("user_reference_alignment.jsonl", [dict(base, reference_id="synthetic_reference", source_action="cash_received",
            source_status="transcribed", source_observation="Synthetic action.", model_support_status="not_recovered_by_model")])
        self.write("moments.jsonl", [])
        self.write("review_queue.jsonl", [])
        objects = self.root / "data/autolabel/objects_v1"
        (objects / "images").mkdir(parents=True)
        image = objects / "images/frame_0000.jpg"
        Image.new("RGB", (40, 60), (80, 100, 120)).save(image)
        self.image_row = dict(image_id="frame_0000", image="images/frame_0000.jpg", image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
            video=base["video"], group_id=base["group_id"], source_sha256=self.video_sha, split="train", timestamp=3, width=40, height=60)
        (objects / "object_review_manifest.jsonl").write_text(json.dumps(self.image_row) + "\n")
        (objects / "summary.json").write_text(json.dumps({"protected_inputs": snapshot(self.root)}))
        self.human = HumanReview(self.root)
        self.objects = ObjectReview(self.root)
        self.store = ReviewStore(self.root / "data/review/event_review.csv")
        parent = make_handler(self.store, self.report / "previews", source=self.root / "data/raw/open", human=self.human, objects=self.objects)
        class QuietHandler(parent):
            def log_message(self, *_):
                pass
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), QuietHandler)
        self.thread = threading.Thread(target=lambda: self.server.serve_forever(poll_interval=.01), daemon=True)
        self.thread.start()
        self.origin = "http://127.0.0.1:" + str(self.server.server_port)

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.temporary.cleanup()

    def write(self, name, rows):
        (self.report / name).write_text("".join(json.dumps(row) + "\n" for row in rows))

    def request(self, method, path, payload=None, headers=None, body=None):
        connection = http.client.HTTPConnection(*self.server.server_address, timeout=5)
        all_headers = {"Origin": self.origin, "Content-Type": "application/json"}
        all_headers.update(headers or {})
        if body is None and payload is not None:
            body = json.dumps(payload).encode()
        try:
            connection.request(method, path, body=body, headers=all_headers)
            response = connection.getresponse()
            raw = response.read()
            return response.status, dict(response.getheaders()), raw
        finally:
            connection.close()

    def json_request(self, method, path, payload=None, **kwargs):
        status, headers, raw = self.request(method, path, payload, **kwargs)
        return status, headers, json.loads(raw)

    def payload(self, **changes):
        task = self.human.catalog()["tasks"][0]
        value = dict(task_id=task["id"], source_fingerprint=task["source_fingerprint"], expected_version=0,
            reviewer="Synthetic reviewer", decision="accept", start_sec=2, end_sec=5,
            observed_action="cash_received", objects=["cash"], note="Synthetic confirmation.",
            evidence_time_sec=3, visibility="clear", event_assessment="not_assessed", violation_type=None)
        value.update(changes)
        return value

    def object_payload(self, **changes):
        value = dict(image_id=self.image_row["image_id"], image_sha256=self.image_row["image_sha256"], expected_version=0,
            reviewer="Synthetic reviewer", review_status="complete", image_assessment="targets_annotated", note="",
            boxes=[dict(class_="cash", visibility="clear", xyxy=[3, 5, 15, 20])])
        value["boxes"][0]["class"] = value["boxes"][0].pop("class_")
        value.update(changes)
        return value

    def test_label_pages_and_catalogs_are_served_without_frontend_proxy(self):
        for path in ("/label", "/label/", "/label/objects", "/label/objects/"):
            with self.subTest(path=path):
                status, headers, raw = self.request("GET", path)
                self.assertEqual(status, 200)
                self.assertIn("text/html", headers["Content-Type"])
                self.assertIn(b"<html", raw.lower())
        status, headers, data = self.json_request("GET", "/api/human-review")
        self.assertEqual(status, 200)
        self.assertEqual(headers["Cache-Control"], "no-store")
        self.assertEqual(data["counts"]["total"], 1)
        status, _, data = self.json_request("GET", "/api/human-objects")
        self.assertEqual(status, 200)
        self.assertEqual(len(data["rows"]), 1)

    def test_save_reload_export_and_conflict_are_actual_http_round_trip(self):
        payload = self.payload()
        status, _, saved = self.json_request("POST", "/api/human-review", payload)
        self.assertEqual(status, 200)
        self.assertEqual(saved["decision"]["version"], 1)
        self.assertEqual(saved["counts"]["confirmed_actions"], 1)
        status, _, catalog = self.json_request("GET", "/api/human-review")
        self.assertEqual(catalog["decisions"][payload["task_id"]]["version"], 1)
        status, _, error = self.json_request("POST", "/api/human-review", payload)
        self.assertEqual(status, 409)
        self.assertIn("error", error)
        status, headers, raw = self.request("GET", "/api/human-review/export")
        self.assertEqual(status, 200)
        self.assertIn("attachment", headers["Content-Disposition"])
        self.assertIn("application/x-ndjson", headers["Content-Type"])
        rows = [json.loads(line) for line in raw.splitlines()]
        self.assertEqual(len(rows), 1)
        self.assertFalse(rows[0]["eligible_for_violation_training"])

    def test_changed_source_returns_409_and_keeps_review_empty(self):
        payload = self.payload()
        with (self.report / "moments.jsonl").open("a") as handle:
            handle.write("\n")
        status, _, _ = self.json_request("POST", "/api/human-review", payload)
        self.assertEqual(status, 409)
        self.assertEqual(self.human.catalog()["counts"]["reviewed"], 0)

    def test_origin_is_required_for_all_mutation_routes(self):
        routes = ("/api/human-review", "/api/human-review/tasks", "/api/human-review/dataset", "/api/human-objects", "/api/human-objects/dataset")
        for path in routes:
            for origin in ("", "https://unrelated.example", "null", "http://["):
                with self.subTest(path=path, origin=origin):
                    status, _, _ = self.json_request("POST", path, {}, headers={"Origin": origin})
                    self.assertEqual(status, 403)
        self.assertEqual(self.human.catalog()["counts"]["reviewed"], 0)
        self.assertEqual(self.objects.catalog()["counts"]["reviewed"], 0)

    def test_malformed_json_size_and_transfer_encoding_never_mutate(self):
        cases = [({"Content-Length": "0"}, b""), ({"Content-Length": "-1"}, b""),
            ({"Content-Length": str(128 * 1024 + 1)}, b""), ({"Content-Length": "invalid"}, b""),
            ({"Transfer-Encoding": "chunked"}, b"0\r\n\r\n"), ({}, b"not json"), ({}, b"[]")]
        for headers, body in cases:
            with self.subTest(headers=headers, body=body):
                status, _, data = self.json_request("POST", "/api/human-review", headers=headers, body=body)
                self.assertEqual(status, 422)
                self.assertIn("error", data)
        self.assertEqual(self.human.catalog()["counts"]["reviewed"], 0)

    def test_short_body_with_valid_json_does_not_save(self):
        raw = json.dumps(self.payload()).encode()
        connection = http.client.HTTPConnection(*self.server.server_address, timeout=5)
        try:
            connection.request("POST", "/api/human-review", raw,
                {"Origin": self.origin, "Content-Type": "application/json", "Content-Length": str(len(raw) + 20)})
            connection.sock.shutdown(socket.SHUT_WR)
            response = connection.getresponse()
            self.assertEqual(response.status, 422)
            response.read()
        finally:
            connection.close()
        self.assertEqual(self.human.catalog()["counts"]["reviewed"], 0)

    def test_manual_task_creation_requires_separate_confirmation(self):
        task = self.human.catalog()["tasks"][0]
        status, _, result = self.json_request("POST", "/api/human-review/tasks", {
            "base_task_id": task["id"], "source_fingerprint": task["source_fingerprint"],
            "reviewer": "Synthetic reviewer", "start_sec": 6, "end_sec": 8, "description": "Missed action."})
        self.assertEqual(status, 201)
        self.assertEqual(result["task"]["source_kind"], "human_added")
        self.assertEqual(result["counts"]["confirmed_actions"], 0)
        status, _, error = self.json_request("POST", "/api/human-review/tasks", {
            "base_task_id": "validation_injection", "source_fingerprint": task["source_fingerprint"],
            "reviewer": "Synthetic reviewer", "start_sec": 0, "end_sec": 2})
        self.assertEqual(status, 404)

    def test_dataset_post_downloads_new_nested_archive_and_no_source_video(self):
        self.assertEqual(self.json_request("POST", "/api/human-review", self.payload())[0], 200)
        status, _, result = self.json_request("POST", "/api/human-review/dataset", {})
        self.assertEqual(status, 201)
        self.assertEqual(result["count"], 1)
        self.assertFalse(result["training_started"])
        export_id = result["export_id"]
        archive_path = self.root / "data/human_review/exports" / export_id / "dataset.zip"
        self.assertTrue(archive_path.is_file())
        status, headers, raw = self.request("GET", f"/api/human-datasets/{export_id}.zip")
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "application/zip")
        self.assertIn("attachment", headers["Content-Disposition"])
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            self.assertEqual(set(archive.namelist()), {"manifest.jsonl", "metadata.json", "label_map.json"})
            record = json.loads(archive.read("manifest.jsonl"))
            self.assertEqual(record["label_scope"], "observed_action")
            self.assertEqual(record["split"], "train")

    def test_archive_download_rejects_traversal_and_symlink_parent(self):
        outside = self.root / "outside_export"
        outside.mkdir()
        (outside / "dataset.zip").write_bytes(b"must not be served")
        exports = self.root / "data/human_review/exports"
        exports.mkdir(exist_ok=True)
        (exports / "escape_test").symlink_to(outside, target_is_directory=True)
        for path in ("/api/human-datasets/escape_test.zip", "/api/human-datasets/../../outside_export/dataset.zip", "/api/human-datasets/%2e%2e%2foutside_export.zip", "/api/human-datasets/missing_export.zip"):
            with self.subTest(path=path):
                status, _, _ = self.request("GET", path)
                self.assertEqual(status, 404)

    def test_object_save_cas_export_and_changed_source(self):
        payload = self.object_payload()
        status, _, result = self.json_request("POST", "/api/human-objects", payload)
        self.assertEqual(status, 200)
        self.assertEqual(result["counts"]["human_boxes"], 1)
        self.assertEqual(self.json_request("POST", "/api/human-objects", payload)[0], 409)
        status, _, raw = self.request("GET", "/api/human-objects/export")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(raw)["version"], 1)
        (self.root / "data/transcripts/observations.jsonl").write_text("changed\n")
        status, _, result = self.json_request("POST", "/api/human-objects", dict(payload, expected_version=1))
        self.assertEqual(status, 409)

    def test_object_dataset_export_keeps_coco_coordinates(self):
        self.assertEqual(self.json_request("POST", "/api/human-objects", self.object_payload())[0], 200)
        status, _, result = self.json_request("POST", "/api/human-objects/dataset", {})
        self.assertEqual(status, 201)
        self.assertEqual(result["training_frames"], 1)
        self.assertFalse(result["independent_validation_set"])
        coco = json.loads((self.root / result["path"] / "coco_train.json").read_text())
        self.assertEqual(coco["annotations"][0]["bbox"], [3, 5, 12, 15])


if __name__ == "__main__":
    unittest.main()
