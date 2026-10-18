import unittest

from .forced_choice_eval import content_for_frames, digit_token_ids, label_from_logits, metrics


class Tokenizer:
    def __call__(self, text, **kwargs):
        return {"input_ids": [ord(c) for c in text]}

    def decode(self, ids, **kwargs):
        return "".join(chr(i) for i in ids)


class ForcedChoiceTests(unittest.TestCase):
    def test_digit_ids_and_boundary_are_verified(self):
        self.assertEqual(digit_token_ids(Tokenizer(), "model\n"), {"0": 48, "1": 49})
        class MergingTokenizer(Tokenizer):
            def __call__(self, text, **kwargs):
                if text == "model\n1":
                    return {"input_ids": [123]}
                return super().__call__(text, **kwargs)
        with self.assertRaisesRegex(ValueError, "boundary"):
            digit_token_ids(MergingTokenizer(), "model\n")

    def test_prompt_uses_only_times_not_private_answers(self):
        frames = [{"timestamp": i, "organizer_answer": "SECRET", "filename": "SECRET.mov", "label": 1}
                  for i in range(16)]
        content = content_for_frames(frames)
        self.assertEqual(sum(c["type"] == "image" for c in content), 16)
        self.assertNotIn("SECRET", str(content))
        self.assertIn("15.000 seconds", str(content))
        with self.assertRaises(ValueError):
            content_for_frames(frames[:4])

    def test_logit_comparison_does_not_use_probability_threshold(self):
        self.assertEqual(label_from_logits({"0": -20., "1": -19.}), 1)
        self.assertEqual(label_from_logits({"0": 100., "1": 100.}), 0)
        with self.assertRaises(ValueError):
            label_from_logits({"0": float("nan"), "1": 0.})

    def test_missing_predictions_remain_abstentions_in_all_scheduled_groups(self):
        samples = [{"group_id": "a", "video": "a", "split": "train", "reference": {"label": 1}},
                   {"group_id": "b", "video": "b", "split": "train", "reference": {"label": 0}}]
        result = metrics(samples, [{"group_id": "a", "condition": "base_same_prompt",
                                    "predicted_label": 1, "decision_scope": "whole_video"}])
        baseline = result["conditions"]["base_same_prompt"]
        self.assertEqual(baseline["accuracy_on_answered"], 1.)
        self.assertEqual(baseline["correct_fraction_all_groups"], .5)
        self.assertEqual(baseline["decision_coverage"], .5)
        self.assertEqual(result["conditions"]["lora_same_prompt"]["answered"], 0)
        self.assertTrue(result["additional_protocol_not_replacement"])


if __name__ == "__main__":
    unittest.main()
