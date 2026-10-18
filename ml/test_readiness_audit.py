import unittest

from readiness_audit import binary_metrics, evaluate_events, validate_full_video_windows, validate_predictions, wilson_interval


class ReadinessMetricTests(unittest.TestCase):
    def test_all_positive_baseline_keeps_accuracy_distinct_from_f1(self):
        result = binary_metrics([0, 0, 0, 0, 1, 1, 1, 1], [1] * 8)
        self.assertEqual(result["accuracy"], 0.5)
        self.assertEqual(result["precision"], 0.5)
        self.assertEqual(result["recall"], 1)
        self.assertAlmostEqual(result["f1"], 2 / 3)
        self.assertEqual(result["false_positive_rate"], 1)

    def test_missing_evaluation_is_not_zero_quality(self):
        result = binary_metrics([], [])
        self.assertIsNone(result["accuracy"])
        self.assertIsNone(result["precision"])
        self.assertIsNone(result["f1"])

    def test_wilson_small_sample_is_wide(self):
        low, high = wilson_interval(7, 8)
        self.assertAlmostEqual(low, 0.5291118178)
        self.assertAlmostEqual(high, 0.9775825085)
        self.assertIsNone(wilson_interval(0, 0))
        with self.assertRaises(ValueError):
            wilson_interval(9, 8)

    def test_prediction_coverage_cannot_silently_drop_hard_video(self):
        with self.assertRaises(ValueError):
            validate_predictions([dict(video="a", target=1, score=0.8)], {"a": 1, "b": 0}, 0.5)

    def test_duplicate_and_mislabeled_predictions_rejected(self):
        row = dict(video="a", target=1, score=0.8)
        with self.assertRaises(ValueError):
            validate_predictions([row, row], {"a": 1}, 0.5)
        with self.assertRaises(ValueError):
            validate_predictions([row], {"a": 0}, 0.5)
        with self.assertRaises(ValueError):
            validate_predictions([dict(row, score=float("nan"))], {"a": 1}, 0.5)

    def test_whole_video_score_cannot_hide_skipped_windows(self):
        spans = [dict(start_sec=0, end_sec=8), dict(start_sec=4, end_sec=12)]
        windows = [dict(spans[0], score=0.2), dict(spans[1], score=0.8)]
        row = dict(windows=windows, window_count=2, score=0.8)
        validate_full_video_windows(row, spans)
        with self.assertRaises(ValueError):
            validate_full_video_windows(dict(row, windows=windows[:1], window_count=1), spans)
        with self.assertRaises(ValueError):
            validate_full_video_windows(dict(row, score=0.2), spans)
        with self.assertRaises(ValueError):
            validate_full_video_windows(dict(row, windows=[dict(windows[0], start_sec=1), windows[1]]), spans)


def event(start, end, kind="cash_left", confirmed=True):
    return dict(start_sec=start, end_sec=end, type=kind, human_confirmed=confirmed)


def reviewed(video, events, complete=True):
    return dict(video=video, human_confirmed=True, coverage_complete=complete, events=events)


class EventGroundTruthTests(unittest.TestCase):
    def test_unavailable_without_human_confirmed_full_coverage(self):
        for truth in (
            {"videos": []},
            {"videos": [reviewed("a", [event(1, 3, confirmed=False)])]},
            {"videos": [reviewed("a", [event(1, 3)], complete=False)]},
        ):
            result = evaluate_events(truth, {"videos": []})
            self.assertEqual(result["status"], "unavailable")
            self.assertIsNone(result["metrics"])

    def test_repeated_detection_is_false_positive_and_wrong_type_not_correct(self):
        truth = {"videos": [reviewed("a", [event(1, 3)]), reviewed("b", [])]}
        predicted = {"videos": [reviewed("a", [event(1, 3), event(1, 3, "key_left")]), reviewed("b", [event(2, 4)])]}
        metrics = evaluate_events(truth, predicted)["metrics"]
        self.assertEqual(metrics["event_confusion"], dict(tp=1, fp=2, fn=0))
        self.assertEqual(metrics["false_positives_per_video"], 1)
        self.assertAlmostEqual(metrics["event_precision"], 1 / 3)
        self.assertEqual(metrics["video_verdict"]["accuracy"], 0.5)
        self.assertEqual(metrics["types"]["key_left"]["f1"], 0)

    def test_matching_is_one_to_one_maximum_cardinality(self):
        # First prediction can match either truth, second can match only the first.
        truth = {"videos": [reviewed("a", [event(0, 4), event(3, 7)])]}
        predicted = {"videos": [reviewed("a", [event(1, 6), event(0, 2)])]}
        result = evaluate_events(truth, predicted, iou_threshold=0.4)
        self.assertEqual(result["metrics"]["event_confusion"], dict(tp=2, fp=0, fn=0))

    def test_missed_events_count_even_with_no_matches_and_no_timing_error(self):
        truth = {"videos": [reviewed("a", [event(1, 3)])]}
        predicted = {"videos": [reviewed("a", [])]}
        metrics = evaluate_events(truth, predicted)["metrics"]
        self.assertEqual(metrics["event_recall"], 0)
        self.assertEqual(metrics["event_f1"], 0)
        self.assertIsNone(metrics["matched_boundary_mean_absolute_error_sec"])

    def test_unknown_types_and_unreviewed_videos_are_not_silently_scored(self):
        truth = {"videos": [reviewed("a", [event(1, 3, "violation_unspecified")])]}
        result = evaluate_events(truth, {"videos": [reviewed("a", [event(1, 3)])]})
        self.assertIsNone(result["metrics"]["type_macro_f1"])
        with self.assertRaises(ValueError):
            evaluate_events(truth, {"videos": [reviewed("b", [])]})
        with self.assertRaises(ValueError):
            evaluate_events(truth, {"videos": []})

    def test_invalid_time_is_not_repaired_into_ground_truth(self):
        with self.assertRaises(ValueError):
            evaluate_events({"videos": [reviewed("a", [event(5, 3)])]}, {"videos": []})


if __name__ == "__main__":
    unittest.main()
