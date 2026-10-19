"""
Unit Tests for Sequential Covering (Rule Learning)
==================================================
Tests rule induction, coverage, prediction, unmatched examples,
and graceful handling of empty datasets.
"""

import unittest
from src.sequential_covering import (
    SequentialCovering,
    Rule,
    CATEGORICAL_ATTRIBUTES,
    CATEGORICAL_CROP_DATASET
)


class TestSequentialCovering(unittest.TestCase):

    def setUp(self):
        self.attributes = CATEGORICAL_ATTRIBUTES
        self.dataset = CATEGORICAL_CROP_DATASET
        self.sc = SequentialCovering(self.attributes)

    def test_learn_rules_for_target_class(self):
        """Test rule learning for a single target class (Rice)."""
        rules = self.sc.fit_target_class(self.dataset, target_class="Rice")
        self.assertGreater(len(rules), 0)

        # All learned rules must predict Rice and have high precision
        for r in rules:
            self.assertEqual(r.target_class, "Rice")
            self.assertEqual(r.precision, 1.0)
            self.assertEqual(r.neg_covered_count, 0)
            self.assertGreater(r.pos_covered_count, 0)

        # Combined coverage must cover all 4 Rice instances in dataset
        covered_ids = set()
        for r in rules:
            covered_ids.update(r.covered_instance_ids)
        self.assertEqual(covered_ids, {1, 2, 3, 4})

    def test_multi_class_rule_learning(self):
        """Test multi-class rule induction across all crop classes."""
        all_rules = self.sc.fit_all_classes(self.dataset, default_class="Rice")
        self.assertGreater(len(all_rules), 1)

        # The last rule should be the default fallback rule (IF True)
        default_rule = all_rules[-1]
        self.assertEqual(len(default_rule.conditions), 0)
        self.assertEqual(default_rule.target_class, "Rice")

    def test_rule_matching(self):
        """Test matching logic for a single IF-THEN rule."""
        rule = Rule({"MoistureLevel": "High", "RainfallLevel": "High"}, target_class="Rice")

        matching_instance = {"MoistureLevel": "High", "RainfallLevel": "High", "NutrientProfile": "Balanced"}
        non_matching_instance = {"MoistureLevel": "Low", "RainfallLevel": "High", "NutrientProfile": "Balanced"}

        self.assertTrue(rule.matches(matching_instance))
        self.assertFalse(rule.matches(non_matching_instance))

    def test_unmatched_example_handling(self):
        """Test graceful handling when an example satisfies none of the learned rules."""
        # Restrict rule set to only Chickpea rules (MoistureLevel = Low)
        chickpea_rules = self.sc.fit_target_class(self.dataset, target_class="Chickpea")

        # An instance with High moisture will not match any Chickpea rule
        wet_instance = {
            "NutrientProfile": "Balanced",
            "MoistureLevel": "High",
            "RainfallLevel": "High",
            "TemperatureLevel": "Warm"
        }

        pred_class, matched_rule = self.sc.predict(wet_instance, rules=chickpea_rules, fallback_class="No Match Found")
        self.assertEqual(pred_class, "No Match Found")
        self.assertIsNone(matched_rule)

    def test_empty_dataset_graceful_handling(self):
        """Test that Sequential Covering does not raise an exception on empty data."""
        empty_rules = self.sc.fit_target_class([], target_class="Rice")
        self.assertEqual(empty_rules, [])

        eval_res = self.sc.evaluate_training_set([], empty_rules)
        self.assertEqual(eval_res["accuracy"], 0.0)
        self.assertEqual(eval_res["total_samples"], 0)

        pred_class, matched_rule = self.sc.predict({"MoistureLevel": "High"}, rules=[], fallback_class="Fallback")
        self.assertEqual(pred_class, "Fallback")
        self.assertIsNone(matched_rule)

    def test_evaluate_training_set(self):
        """Test calculation of training set coverage and accuracy."""
        all_rules = self.sc.fit_all_classes(self.dataset)
        metrics = self.sc.evaluate_training_set(self.dataset, all_rules)

        self.assertEqual(metrics["total_samples"], len(self.dataset))
        self.assertGreaterEqual(metrics["accuracy"], 90.0)
        self.assertIn("Training Set Metrics", metrics["metric_type"])


if __name__ == "__main__":
    unittest.main()
