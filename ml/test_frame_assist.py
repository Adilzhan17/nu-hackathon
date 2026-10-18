import copy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest import mock

from PIL import Image

from frame_assist import AssistError, FrameAssist, MODEL, extract_frame, image_inputs, normalize_result


class SyntheticHuman:
    def __init__(self, task): self.task = task
    def catalog(self): return {"tasks": [self.task]}


class SyntheticObjects:
    def __init__(self, root):
        self.source = root / "objects"
        self.source.mkdir()
        path = self.source / "frame.jpg"
        Image.new("RGB", (400, 300), (70, 100, 130)).save(path, quality=95)
        self.rows = {"frame_0": dict(image_id="frame_0", image="frame.jpg", width=400, height=300,
            image_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), video="videos/synthetic.mov",
            source_sha256="a" * 64, group_id="synthetic_group", split="train", timestamp=1)}
    def check_sources(self): pass
    def image_path(self, row): return self.source / row["image"]


def model_result(**changes):
    value = dict(description="Synthetic scene.", visible_facts=["Synthetic rectangle."], context_hypotheses=[],
        context_disagreements=[], uncertainties=["Approximate box."], questions=["Check object class."],
        candidates=[dict(image_index=0, **{"class": "cash"}, bbox=[.1, .2, .3, .4],
                         description="Synthetic target.", visibility="clear")])
    value.update(changes)
    return value


def response(value=None):
    return dict(status="completed", usage=dict(input_tokens=25, output_tokens=20, total_tokens=45),
                output=[dict(type="message", content=[dict(type="output_text", text=json.dumps(value or model_result()))])])


class FrameAssistTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.objects = SyntheticObjects(self.root)
        self.human = SyntheticHuman(dict(id="task", source_fingerprint="source1", video="videos/synthetic.mov",
            video_sha256="a" * 64, group_id="synthetic_group", duration_sec=10, split="train"))
        self.service = FrameAssist(self.root, self.human, self.objects)
        self.key_mock = mock.patch.object(self.service, "_key", return_value="synthetic-test-key")
        self.key_mock.start()

    def tearDown(self):
        deadline = time.time() + 4
        while self.service.status()["busy"] and time.time() < deadline: time.sleep(.01)
        self.key_mock.stop()
        self.temp.cleanup()

    def payload(self, **changes):
        return dict(image_id="frame_0", image_sha256=self.objects.rows["frame_0"]["image_sha256"], **changes)

    def wait(self, job_id):
        deadline = time.time() + 4
        while time.time() < deadline:
            result = self.service.get(job_id)
            if result["status"] in {"completed", "failed"}: return result
            time.sleep(.01)
        self.fail("Synthetic job did not complete")

    def test_full_scene_overlapping_tiles_and_roi_native_coordinates(self):
        image = Image.new("RGB", (4000, 3000))
        tiles = image_inputs(image)
        self.assertEqual(len(tiles), 5)
        self.assertEqual(tiles[0]["width"], 1536)
        self.assertTrue(all(max(t["width"], t["height"]) <= 1024 for t in tiles[1:]))
        self.assertLess(tiles[2]["bounds"][0], tiles[1]["bounds"][2])
        roi = image_inputs(image, [100, 200, 500, 800])
        parsed = normalize_result(model_result(candidates=[dict(image_index=1, **{"class": "key"},
            bbox=[.25, .25, .75, .75], description="Synthetic key.", visibility="uncertain")]), roi, 4000, 3000)
        self.assertEqual(parsed["candidates"][0]["xyxy"], [200, 350, 400, 650])
        self.assertFalse(parsed["eligible_for_training"])
        self.assertFalse(parsed["candidates"][0]["eligible_for_training"])
        for bad in [[0, 0, 5000, 100], [0, 0, 1, 1], [0, float("nan"), 5, 5], [True, 0, 5, 5]]:
            with self.subTest(bad=bad), self.assertRaises(AssistError): image_inputs(image, bad)

    def test_invalid_model_boxes_are_never_saved_as_proposals(self):
        images = image_inputs(Image.new("RGB", (400, 300)))
        original = model_result()
        for field, value in [("bbox", [0, 0, 1.01, 1]), ("bbox", [0, 0, float("nan"), 1]),
                             ("bbox", [0, 0, 0, 1]), ("image_index", 9), ("class", "invented")]:
            data = copy.deepcopy(original)
            data["candidates"][0][field] = value
            with self.subTest(field=field), self.assertRaises(AssistError): normalize_result(data, images, 400, 300)

    def test_success_cache_no_additional_paid_call_and_native_frame(self):
        with mock.patch.object(self.service, "_call_api", return_value=response()) as api:
            first = self.service.start(self.payload())
            result = self.wait(first["job_id"])
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["usage"]["total_tokens"], 45)
            self.assertEqual(result["result"]["candidates"][0]["xyxy"], [40, 60, 120, 120])
            self.assertEqual(api.call_args.args[0]["model"], MODEL)
            self.assertFalse(api.call_args.args[0]["store"])
            self.assertTrue(api.call_args.args[0]["text"]["format"]["strict"])
            self.assertEqual(api.call_args.args[0]["reasoning"], {"effort": "medium"})
            self.assertEqual(len([c for c in api.call_args.args[0]["input"][0]["content"] if c["type"] == "input_image"]), 5)
            with Image.open(self.service.frame_path(first["job_id"])) as image:
                self.assertEqual(image.size, (400, 300))
            second = self.service.start(self.payload())
            self.assertTrue(second["cache_hit"])
            self.assertEqual(api.call_count, 1)
            reopened = FrameAssist(self.root, self.human, self.objects)
            with mock.patch.object(reopened, "_key", return_value="fake"), mock.patch.object(reopened, "_call_api") as other:
                self.assertTrue(reopened.start(self.payload())["cache_hit"])
                other.assert_not_called()
            self.assertNotIn("synthetic-test-key", json.dumps(result))

    def test_changed_context_invalidates_cache_and_remains_weak(self):
        report = self.root / "data/autolabel/detailed_train_report"
        report.mkdir(parents=True)
        moment = dict(group_id="synthetic_group", start_sec=0, end_sec=2, observed_action="cash_received",
            objects=["cash"], proposal_ids=["p1"], uncertainty_reasons=["Synthetic uncertainty"])
        (report / "moments.jsonl").write_text(json.dumps(moment) + "\n")
        (report / "event_details.jsonl").write_text(json.dumps(dict(proposal_id="p1", evidence_description="Synthetic Qwen context")) + "\n")
        with mock.patch.object(self.service, "_call_api", return_value=response()) as api:
            self.wait(self.service.start(self.payload())["job_id"])
            context = json.loads(api.call_args.args[0]["input"][0]["content"][0]["text"])
            self.assertFalse(context["context"][0]["confirmed"])
            self.assertIn("Synthetic Qwen context", context["context"][0]["descriptions"])
            moment["objects"] = ["key"]
            (report / "moments.jsonl").write_text(json.dumps(moment) + "\n")
            self.wait(self.service.start(self.payload())["job_id"])
            self.assertEqual(api.call_count, 2)

    def test_stale_task_and_frame_do_not_contact_provider(self):
        with mock.patch.object(self.service, "_call_api") as api:
            for payload in [dict(task_id="task", source_fingerprint="old", timestamp_sec=1),
                            dict(image_id="frame_0", image_sha256="wrong"),
                            dict(image_id="missing", image_sha256="wrong"), self.payload(path="/etc/passwd")]:
                with self.subTest(payload=payload), self.assertRaises(AssistError): self.service.start(payload)
            api.assert_not_called()

    def test_qwen_window_context_survives_without_event_proposals(self):
        report = self.root / "data/autolabel/detailed_train_report"
        report.mkdir(parents=True)
        window = dict(start_sec=0, end_sec=5, sequence_summary="Synthetic scene context.", inference_status="completed")
        distant = dict(start_sec=80, end_sec=85, sequence_summary="Distant context must be excluded.")
        (report / "video_timelines.jsonl").write_text(json.dumps(dict(group_id="synthetic_group", windows=[window, distant])) + "\n")
        with mock.patch.object(self.service, "_call_api", return_value=response()) as api:
            self.wait(self.service.start(self.payload())["job_id"])
            context = json.loads(api.call_args.args[0]["input"][0]["content"][0]["text"])["context"]
            self.assertEqual(len(context), 1)
            self.assertEqual(context[0]["sequence_summary"], "Synthetic scene context.")
            self.assertFalse(context[0]["confirmed"])

    def test_http_errors_never_expose_provider_body_or_authorization(self):
        import io
        import urllib.error
        for status, expected in [(401, "provider_auth"), (403, "provider_auth"), (404, "model_unavailable"),
                                 (429, "provider_limit"), (500, "provider_error")]:
            error = urllib.error.HTTPError("https://api.openai.com/v1/responses", status,
                "synthetic-secret-provider-error", {}, io.BytesIO(b"synthetic-secret-provider-body"))
            with mock.patch("frame_assist.urllib.request.urlopen", side_effect=error) as call:
                with self.assertRaises(AssistError) as caught: self.service._call_api({})
                self.assertEqual(caught.exception.code, expected)
                self.assertNotIn("synthetic-secret", str(caught.exception))
                self.assertEqual(call.call_count, 1)

    def test_concurrent_identical_request_reuses_job_others_wait(self):
        release = threading.Event()
        def call(payload):
            release.wait(3)
            return response()
        with mock.patch.object(self.service, "_call_api", side_effect=call) as api:
            first = self.service.start(self.payload())
            try:
                second = self.service.start(self.payload())
                self.assertEqual(first["job_id"], second["job_id"])
                with self.assertRaises(AssistError) as caught: self.service.start(self.payload(roi=[0, 0, 100, 100]))
                self.assertEqual(caught.exception.status_code, 429)
            finally: release.set()
            self.assertEqual(self.wait(first["job_id"])["status"], "completed")
            self.assertEqual(api.call_count, 1)

    def test_failure_refusal_and_incomplete_are_not_retried_or_leaked(self):
        failures = [RuntimeError("synthetic-secret-never-output"),
            dict(status="completed", output=[dict(content=[dict(type="refusal", refusal="private provider text")])]),
            dict(status="incomplete", output=[])]
        for index, failure in enumerate(failures):
            method = {"side_effect": failure} if isinstance(failure, Exception) else {"return_value": failure}
            with mock.patch.object(self.service, "_call_api", **method) as api:
                result = self.wait(self.service.start(self.payload(roi=[index, 0, 100, 100]))["job_id"])
                self.assertEqual(result["status"], "failed")
                self.assertIsNone(result["result"])
                self.assertEqual(api.call_count, 1)
                self.assertNotIn("synthetic-secret-never-output", json.dumps(result))
                self.assertNotIn("private provider text", json.dumps(result))

    def test_freeze_only_never_calls_api(self):
        with mock.patch.object(self.service, "_call_api") as api:
            metadata = self.service.prepare_frame(self.payload())
            self.assertTrue(metadata["path"].is_relative_to(self.root / "data/human_review"))
            self.assertEqual(metadata["source"]["image_id"], "frame_0")
            self.assertEqual(metadata["timestamp_sec"], 1)
            api.assert_not_called()

    def test_nonzero_pts_are_relative_and_source_verified(self):
        import av
        path = self.root / "data/raw/open/videos/synthetic.mov"
        path.parent.mkdir(parents=True)
        with av.open(str(path), "w") as output:
            stream = output.add_stream("libx264", rate=10)
            stream.width, stream.height = 64, 48
            stream.pix_fmt = "yuv420p"
            stream.time_base = Fraction(1, 10)
            for i in range(20):
                frame = av.VideoFrame.from_image(Image.new("RGB", (64, 48), (i * 8, 40, 70)))
                frame.pts, frame.time_base = 90 + i, Fraction(1, 10)
                for packet in stream.encode(frame): output.mux(packet)
            for packet in stream.encode(): output.mux(packet)
        self.human.task["video_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        image, actual = extract_frame(path, .76)
        self.assertEqual(image.size, (64, 48))
        self.assertAlmostEqual(actual, .8, places=3)
        metadata = self.service.prepare_frame(dict(task_id="task", source_fingerprint="source1", timestamp_sec=.76))
        self.assertAlmostEqual(metadata["timestamp_sec"], .8, places=3)
        path.write_bytes(path.read_bytes() + b"changed")
        with self.assertRaises(AssistError) as caught:
            self.service.prepare_frame(dict(task_id="task", source_fingerprint="source1", timestamp_sec=.76))
        self.assertEqual(caught.exception.code, "stale_source")


if __name__ == "__main__": unittest.main()
