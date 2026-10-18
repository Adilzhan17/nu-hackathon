import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ml.gemma_lab.adaptation.eval_prompts import prompt
from ml.gemma_lab.adaptation.run_eval import (
    ACTION_CODES, blind_sample, metrics, parse_answer, run,
)


class ParserTests(unittest.TestCase):
    def test_tentative_normal_class_is_not_supported_normal_verdict(self):
        raw = json.dumps(dict(predicted_label=0, rule_verdict="insufficient_data",
                              evidence_frames=["f0"], reason="Классификация по выборке, не подтверждение нормы"))
        value = parse_answer(raw, "video", ["f0", "f1"])
        self.assertEqual(value["predicted_label"], 0)
        self.assertEqual(value["rule_verdict"], "insufficient_data")

    def test_binary_prediction_does_not_override_evidence_gate(self):
        raw = json.dumps(dict(predicted_label=1, rule_verdict="violation_supported",
                              evidence_frames=["f0"], reason="Предположение"))
        value = parse_answer(raw, "video", ["f0", "f1"])
        self.assertEqual(value["predicted_label"], 1)
        self.assertEqual(value["rule_verdict"], "insufficient_data")

    def test_invalid_frame_and_truncated_json_abstain(self):
        for raw in ['{"predicted_label": 1', json.dumps(dict(predicted_label=1,
                rule_verdict="violation_supported", evidence_frames=["f0", "f999"], reason="x"))]:
            self.assertIsNone(parse_answer(raw, "video", ["f0", "f1"])["predicted_label"])

    def test_single_complete_outer_fence_only(self):
        raw = '{"observed_action":"cash_counting","description":"Пересчёт","review_status":"proposed"}'
        value = parse_answer('```json\n' + raw + '\n```', "action", ["f0", "f1"])
        self.assertEqual(value["parse_status"], "accepted")
        self.assertTrue(value["wrapper_removed"])
        for malformed in ['```json\n' + raw, 'Here:\n```json\n' + raw + '\n```',
                          '```json\n' + raw + '\n```\n```json\n{}\n```', '```json\n{"observed_action":\n```']:
            self.assertEqual(parse_answer(malformed, "action", ["f0", "f1"])["parse_status"], "invalid_json_or_schema")

    def test_action_matches_training_output_contract(self):
        value = parse_answer('{"observed_action":"cash_counting","description":"Пересчитывает пачку"}', "action", ["f0", "f1"])
        self.assertEqual(value["observed_action"], "cash_counting")
        self.assertTrue(value["legacy_two_field_output"])

    def test_training_review_status_is_accepted_but_human_confirmation_rejected(self):
        response = dict(observed_action="cash_counting", description="Пересчитывает пачку", review_status="proposed")
        value = parse_answer(json.dumps(response), "action", ["f0", "f1"])
        self.assertEqual(value["observed_action"], "cash_counting")
        self.assertEqual(value["review_status"], "proposed")
        self.assertFalse(value["legacy_two_field_output"])
        response["review_status"] = "human_confirmed"
        self.assertEqual(parse_answer(json.dumps(response), "action", ["f0", "f1"])["parse_status"], "invalid_json_or_schema")

    def test_adapter_comparison_uses_exact_same_prompt(self):
        frames = [dict(id="f0", timestamp=0), dict(id="f1", timestamp=1)]
        self.assertEqual(prompt("base_v2prompt", "action", frames, ACTION_CODES),
                         prompt("lora_v2prompt", "action", frames, ACTION_CODES))


class MetricsTests(unittest.TestCase):
    def test_classification_and_supported_verdict_metrics_are_separate(self):
        samples = [dict(sample_id="s", task="video", video="a.mov", group_id="g",
                        partition="inner_holdout", reference=dict(label=0))]
        rows = [dict(sample_id="s", task="video", group_id="g", condition="base_v2prompt",
                     predicted_label=0, decision_scope="whole_video", rule_verdict="insufficient_data")]
        result = metrics(samples, rows, ("base_v2prompt", "lora_v2prompt"))
        base = result["conditions"]["base_v2prompt"]["inner_holdout"]
        self.assertEqual(base["whole_video"]["accuracy_on_answered"], 1)
        self.assertEqual(base["rule_supported_whole_video"]["decision_coverage"], 0)
        self.assertEqual(result["fully_paired_samples"], 0)
        self.assertEqual(result["paired_only_conditions"]["base_v2prompt"], {})


class RunnerSmokeTests(unittest.TestCase):
    def test_fake_backend_runs_paired_windows_without_target_leak_and_resumes(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frames = []
            for i in range(16):
                path = root / f"f{i}.jpg"
                Image.new("RGB", (8, 8), color=(i, 0, 0)).save(path)
                frames.append(dict(path=path.name, timestamp=float(i), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            sample = dict(sample_id="s", group_id="g", video="DO_NOT_SEND_FILENAME.mov", task="video",
                          partition="inner_holdout", duration_sec=16, frames=frames,
                          reference=dict(label=0, description="SECRET_TARGET_TEXT"), note="SECRET_TARGET_TEXT")
            manifest = root / "samples.jsonl"
            manifest.write_text(json.dumps(sample) + "\n")
            protocol = root / "protocol.json"
            protocol.write_text(json.dumps(dict(seed=1, group_membership=dict(g="inner_holdout"))))
            adapter = root / "adapter"
            adapter.mkdir()
            (adapter / "adapter_config.json").write_text("{}")
            (adapter / "adapter_model.safetensors").write_bytes(b"fake test checkpoint")
            args = argparse.Namespace(samples=[manifest], run_root=root, protocol=protocol, model=root,
                adapter=adapter, output=root / "eval", lock=root / "gpu.lock", seconds=60,
                task="video", partition="inner_holdout", conditions=["base_v2prompt", "lora_v2prompt"], resume=False)
            calls = []

            class FakeBackend:
                def provenance(self):
                    return {"test_only": True}

                def predict(self, images, text, **kwargs):
                    calls.append((text, kwargs))
                    assert "SECRET_TARGET_TEXT" not in text
                    assert "DO_NOT_SEND_FILENAME" not in text
                    assert len(images) == 4
                    return json.dumps(dict(predicted_label=0, rule_verdict="insufficient_data",
                                           evidence_frames=[], reason="Иное наблюдение адаптера" if kwargs["adapter_enabled"]
                                           else "Наблюдаемые действия без определённого вывода"))

            with patch("ml.gemma_lab.adaptation.hf_backend.load_backend", return_value=FakeBackend()), \
                 patch("ml.gemma_lab.adaptation.run_eval.subprocess.check_output", return_value=""):
                result = run(args)
                self.assertEqual(result["status"], "completed")
                self.assertEqual(len(calls), 8)
                self.assertTrue(all(call[1]["max_new_tokens"] == 128 for call in calls))
                self.assertEqual([call[1]["adapter_enabled"] for call in calls], [False] * 4 + [True] * 4)
                self.assertEqual([call[0] for call in calls[:4]], [call[0] for call in calls[4:]])
                args.resume = True
                resumed = run(args)
                self.assertEqual(resumed["completed"], 2)
                self.assertEqual(len(calls), 8)
                # Changed input must never silently reuse an old evaluation.
                sample["reference"]["label"] = 1
                manifest.write_text(json.dumps(sample) + "\n")
                with self.assertRaises(ValueError):
                    run(args)


if __name__ == "__main__":
    unittest.main()
