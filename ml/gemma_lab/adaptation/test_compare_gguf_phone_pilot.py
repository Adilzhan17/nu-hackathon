import unittest

from ml.gemma_lab.adaptation.compare_gguf_phone_pilot import summarize, compare


class PhonePilotComparisonTests(unittest.TestCase):
    def test_counter_loading_is_not_relabelled_as_counting(self):
        reference = dict(task_id="loading", observed_action="cash_counter_loading", start_sec=6.16, end_sec=8.17)
        result = dict(observations=[dict(kind="event", observation_id="model:counter", action="cash_counter_used",
                      start_sec=6.2, end_sec=8.0, stage="overview_001", description="Устройство считает")])
        item = summarize(result, [reference])["reference_comparison"][0]
        self.assertFalse(item["scenario_action_code_exists"])
        self.assertEqual(item["exact_action_overview_temporal_matches"], [])
        self.assertEqual(len(item["nearby_visual_proposals"]), 1)

    def test_exact_phone_match_uses_overview_only(self):
        ref = dict(task_id="phone", observed_action="phone_put_away", start_sec=0, end_sec=4)
        event = dict(kind="event", observation_id="phone", action="phone_put_away", start_sec=0, end_sec=4)
        result = dict(observations=[dict(event, stage="detail_000")])
        self.assertEqual(summarize(result, [ref])["reference_comparison"][0]["exact_action_overview_temporal_matches"], [])
        result["observations"][0]["stage"] = "overview_000"
        self.assertEqual(len(summarize(result, [ref])["reference_comparison"][0]["exact_action_overview_temporal_matches"]), 1)

    def test_demo_never_claims_accuracy_or_prompt_only_causality(self):
        result = dict(organizer_answers_in_prompt=False, cloud_ai_used=False,
                      status="completed", observations=[], rules=[], elapsed_sec=10,
                      model_id="test_model", model_revision="test_revision", quantization="test_q4",
                      runtime_release="test_runtime", inference_config={"test": True})
        report = compare(result, dict(result), {}, [])
        self.assertFalse(report["is_accuracy_estimate"])
        self.assertIsNone(report["measured_accuracy"])
        self.assertFalse(report["isolated_prompt_ablation_established"])
        self.assertFalse(report["schema_provenance"]["explicitly_same"])
        with self.assertRaises(ValueError):
            compare(result, dict(result, organizer_answers_in_prompt=True), {}, [])


if __name__ == "__main__":
    unittest.main()
