import copy
import unittest

from ml.gemma_lab.adaptation.eval import (
    classify, evaluate, event_reference_recovery, grouped_train_partition,
    normalize_partition, prediction_from_result, validate_fit_sources, video_targets,
)


def target(group, label, split="train", **kwargs):
    return dict(video=f"{group}.mov", group_id=group, label=label, split=split,
                label_scope="whole_video", **kwargs)


def prediction(group, label, **kwargs):
    return dict(group_id=group, predicted_label=label, decision_scope="whole_video", **kwargs)


class WholeVideoTests(unittest.TestCase):
    def test_abstention_is_not_a_negative_prediction(self):
        targets = [target("p", 1), target("n", 0)]
        result = classify(targets, [prediction("p", None), prediction("n", None)])
        self.assertEqual(result["decision_coverage"], 0)
        self.assertIsNone(result["accuracy_on_answered"])
        self.assertEqual(result["correct_fraction_all_groups"], 0)
        self.assertEqual(result["recall_positive_end_to_end"], 0)
        self.assertIsNone(result["precision_positive"])
        self.assertEqual(result["confusion_matrix"]["true_negative"]["predicted_negative"], 0)

    def test_selective_accuracy_does_not_hide_abstention(self):
        targets = [target("a", 1), target("b", 1), target("c", 0), target("d", 0)]
        report = classify(targets, [prediction("a", 1), prediction("b", None), prediction("c", 1)])
        self.assertEqual(report["decision_coverage"], 0.5)
        self.assertEqual(report["accuracy_on_answered"], 0.5)
        self.assertEqual(report["correct_fraction_all_groups"], 0.25)
        self.assertEqual(report["recall_positive_on_answered"], 1)
        self.assertEqual(report["recall_positive_end_to_end"], 0.5)
        self.assertEqual(report["predictions_present"], 3)
        self.assertEqual(report["run_status_counts"]["no_run"], 1)

    def test_duplicate_predictions_fail_instead_of_choosing_best(self):
        with self.assertRaises(ValueError):
            classify([target("p", 1)], [prediction("p", 0), prediction("p", 1)])

    def test_episode_cannot_become_whole_video_normal(self):
        result = dict(status="completed", rules=[dict(rule_id="phone_before_counting",
                      verdict="no_violation_observed", evidence=[{"start_sec": 1}])])
        self.assertIsNone(prediction_from_result(result, "p")["predicted_label"])
        with self.assertRaises(ValueError):
            classify([target("p", 0)], [dict(group_id="p", predicted_label=0, decision_scope="episode")])

    def test_only_supported_positive_rule_not_raw_claim_is_positive(self):
        result = dict(rules=[], model_rule_claims=[dict(model_verdict="violation_supported")])
        self.assertIsNone(prediction_from_result(result, "p")["predicted_label"])
        result["rules"] = [dict(rule_id="safe_on_departure", verdict="violation_supported", evidence=[])]
        self.assertIsNone(prediction_from_result(result, "p")["predicted_label"])
        result["rules"][0]["evidence"] = [{"observation_id": "e"}]
        self.assertEqual(prediction_from_result(result, "p")["predicted_label"], 1)

    def test_does_not_score_answer_contaminated_predictions(self):
        with self.assertRaises(ValueError):
            classify([target("p", 1)], [prediction("p", 1, organizer_answers_in_prompt=True)])


class PartitionTests(unittest.TestCase):
    def test_partition_deterministic_and_preserves_frozen_split(self):
        targets = [target(str(i), i % 2) for i in range(12)] + [target("v", 1, "val")]
        first = grouped_train_partition(targets)
        self.assertEqual(first, grouped_train_partition(list(reversed(targets))))
        self.assertEqual(first["assignments"]["v"], "reused_val_diagnostic")
        self.assertEqual(first["counts"]["train_tune"], 4)
        with self.assertRaises(ValueError):
            validate_fit_sources(["v"], first)
        valid = [group for group, split in first["assignments"].items() if split == "train_fit"]
        validate_fit_sources(valid, first)
        bad = copy.deepcopy(first)
        bad["assignments"]["v"] = "train_tune"
        with self.assertRaises(ValueError):
            evaluate(targets, [], bad)

    def test_duplicates_and_manifest_windows_count_once(self):
        rows = [target("a", 1, video_sha256="sha"), target("a", 1, video_sha256="sha")]
        rows[1]["video"] = "copy.mov"
        labels = video_targets(rows)
        self.assertEqual(len(labels), 1)
        self.assertIsNone(labels[0]["window_violation_target"])
        self.assertEqual(len(labels[0]["aliases"]), 2)
        rows[1]["split"] = "val"
        with self.assertRaises(ValueError):
            video_targets(rows)

    def test_identical_content_cannot_have_different_groups(self):
        with self.assertRaises(ValueError):
            video_targets([target("a", 1, video_sha256="same"), target("b", 1, "val", video_sha256="same")])

    def test_reuses_audited_group_membership_without_reselection(self):
        protocol = dict(seed=7301, group_membership={"a": "fit", "b": "inner_holdout", "v": "diagnostic_val"})
        partition = normalize_partition(protocol)
        self.assertEqual(partition["assignments"], {"a": "train_fit", "b": "inner_train_holdout", "v": "reused_val_diagnostic"})
        report = evaluate([target("a", 0), target("b", 1), target("v", 0, "val")], [], protocol)
        self.assertEqual(report["whole_video"]["inner_train_holdout"]["total_source_groups"], 1)


class EventTests(unittest.TestCase):
    def reference(self, task="r", start=1, end=3):
        return dict(group_id="g", task_id=task, observed_action="cash_received", start_sec=start,
                    end_sec=end, action_confirmed=True, eligible_for_action_training=True,
                    label_scope="observed_action", boundary_status="human_accepted_interval_not_independently_retimed")

    def test_one_prediction_cannot_recover_multiple_references(self):
        refs = [self.reference("a"), self.reference("b", 3, 5)]
        preds = [dict(group_id="g", action="cash_received", start_sec=1, end_sec=5)]
        report = event_reference_recovery(refs, preds)
        self.assertEqual(report["recovered"], 1)
        self.assertEqual(report["sparse_reference_recovery"], 0.5)
        self.assertFalse(report["unmatched_proposals_are_false_positives"])
        self.assertIn("event_precision", report["unsupported_metrics"])

    def test_only_explicit_human_actions_and_exact_action_types(self):
        refs = [self.reference(), dict(self.reference("b"), action_confirmed=False),
                dict(self.reference("c"), observed_action="cash_visible")]
        preds = [dict(group_id="g", action="cash_placed", start_sec=1, end_sec=3)]
        report = event_reference_recovery(refs, preds)
        self.assertEqual(report["confirmed_references"], 1)
        self.assertEqual(report["recovered"], 0)
        self.assertEqual(len(report["excluded"]), 2)

    def test_negative_or_nonfinite_time_is_rejected(self):
        with self.assertRaises(ValueError):
            event_reference_recovery([self.reference(start=-1)], [])
        with self.assertRaises(ValueError):
            event_reference_recovery([self.reference(end=float("inf"))], [])


if __name__ == "__main__":
    unittest.main()
