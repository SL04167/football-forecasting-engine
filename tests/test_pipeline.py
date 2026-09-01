from __future__ import annotations

import unittest

from tff.data import generate_matches
from tff.features import FEATURE_COLUMNS, build_features
from tff.model import evaluate_models


class ForecastPipelineTests(unittest.TestCase):
    def test_generator_is_deterministic(self) -> None:
        first = generate_matches(120, seed=9)
        second = generate_matches(120, seed=9)
        self.assertTrue(first.equals(second))

    def test_initial_match_contains_only_neutral_history(self) -> None:
        features = build_features(generate_matches(120, seed=3))
        first = features.iloc[0]
        self.assertEqual(first["home_form"], 0.0)
        self.assertEqual(first["away_form"], 0.0)
        self.assertEqual(first["rating_difference"], 0.0)

    def test_feature_frame_is_complete(self) -> None:
        features = build_features(generate_matches(160, seed=12))
        self.assertEqual(len(features), 160)
        self.assertFalse(features[FEATURE_COLUMNS].isna().any().any())

    def test_both_models_return_valid_metrics(self) -> None:
        features = build_features(generate_matches(240, seed=4))
        reports = evaluate_models(features)
        self.assertEqual({report.name for report in reports}, {"logistic_regression", "gradient_boosting"})
        for report in reports:
            self.assertGreaterEqual(report.accuracy, 0.0)
            self.assertLessEqual(report.accuracy, 1.0)
            self.assertGreater(report.log_loss, 0.0)
            self.assertGreater(report.test_matches, 0)


if __name__ == "__main__":
    unittest.main()
