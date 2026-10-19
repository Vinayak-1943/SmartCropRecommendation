"""
Unit Tests for Concept Learning Lab (Find-S and Candidate Elimination)
======================================================================
Tests positive example generalization, negative example specialization,
and version space collapse handling for inconsistent/noisy data.
"""

import unittest
from src.concept_learning import (
    FindS,
    CandidateElimination,
    DEFAULT_ATTRIBUTES,
    DEFAULT_EXAMPLES,
    is_more_general_or_equal,
    matches_instance
)


class TestConceptLearning(unittest.TestCase):

    def setUp(self):
        self.attributes = DEFAULT_ATTRIBUTES
        self.attr_names = list(DEFAULT_ATTRIBUTES.keys())

    def test_find_s_positive_examples(self):
        """Test that Find-S generalizes along matching and varying attributes."""
        find_s = FindS(self.attr_names)
        examples = [
            {"features": {"SoilType": "Clayey", "Climate": "Tropical", "Moisture": "High", "Rainfall": "Heavy"}, "label": "Yes"},
            {"features": {"SoilType": "Clayey", "Climate": "Tropical", "Moisture": "High", "Rainfall": "Moderate"}, "label": "Yes"}
        ]
        h = find_s.fit(examples)
        self.assertEqual(h, ("Clayey", "Tropical", "High", "?"))

    def test_find_s_ignores_negative_examples(self):
        """Test that Find-S completely ignores negative examples."""
        find_s = FindS(self.attr_names)
        examples = [
            {"features": {"SoilType": "Clayey", "Climate": "Tropical", "Moisture": "High", "Rainfall": "Heavy"}, "label": "Yes"},
            {"features": {"SoilType": "Sandy", "Climate": "Arid", "Moisture": "Low", "Rainfall": "Light"}, "label": "No"}
        ]
        h = find_s.fit(examples)
        # Should stay at the first positive example, unaffected by negative
        self.assertEqual(h, ("Clayey", "Tropical", "High", "Heavy"))

    def test_find_s_full_wildcard(self):
        """Test that Find-S generalizes to all wildcards if all attributes differ."""
        find_s = FindS(self.attr_names)
        examples = [
            {"features": {"SoilType": "Clayey", "Climate": "Tropical", "Moisture": "High", "Rainfall": "Heavy"}, "label": "Yes"},
            {"features": {"SoilType": "Sandy", "Climate": "Arid", "Moisture": "Low", "Rainfall": "Light"}, "label": "Yes"}
        ]
        h = find_s.fit(examples)
        self.assertEqual(h, ("?", "?", "?", "?"))

    def test_candidate_elimination_default_dataset(self):
        """Test Candidate Elimination on the 5-example teaching dataset."""
        ce = CandidateElimination(self.attributes)
        s_boundary, g_boundary = ce.fit(DEFAULT_EXAMPLES)

        self.assertFalse(ce.is_version_space_empty)
        # S must be exactly ('Clayey', 'Tropical', '?', '?')
        self.assertEqual(s_boundary, {("Clayey", "Tropical", "?", "?")})
        # G must contain ('Clayey', '?', '?', '?') and ('?', 'Tropical', '?', '?')
        expected_g = {("Clayey", "?", "?", "?"), ("?", "Tropical", "?", "?")}
        self.assertEqual(g_boundary, expected_g)

    def test_candidate_elimination_prediction_agreement(self):
        """Test prediction for unanimous agreement, disagreement, and unanimous rejection."""
        ce = CandidateElimination(self.attributes)
        ce.fit(DEFAULT_EXAMPLES)

        # Unanimous Yes: Clayey, Tropical, Low, Light matches both S and G
        pred_yes = ce.predict({"SoilType": "Clayey", "Climate": "Tropical", "Moisture": "Low", "Rainfall": "Light"})
        self.assertEqual(pred_yes, "Yes (Unanimously Positive)")

        # Unanimous No: Sandy, Arid, Low, Light matches neither S nor G
        pred_no = ce.predict({"SoilType": "Sandy", "Climate": "Arid", "Moisture": "Low", "Rainfall": "Light"})
        self.assertEqual(pred_no, "No (Unanimously Negative)")

        # Boundary Disagreement: Clayey, Temperate, High, Heavy matches one G but not S
        pred_disagree = ce.predict({"SoilType": "Clayey", "Climate": "Temperate", "Moisture": "High", "Rainfall": "Heavy"})
        self.assertEqual(pred_disagree, "Uncertain (Boundary Disagreement)")

    def test_candidate_elimination_inconsistent_data(self):
        """Test that Candidate Elimination detects version space collapse on conflicting examples."""
        ce = CandidateElimination(self.attributes)
        # Two conflicting examples with exact same features but opposite labels
        conflicting_examples = [
            {"features": {"SoilType": "Clayey", "Climate": "Tropical", "Moisture": "High", "Rainfall": "Heavy"}, "label": "Yes"},
            {"features": {"SoilType": "Clayey", "Climate": "Tropical", "Moisture": "High", "Rainfall": "Heavy"}, "label": "No"}
        ]
        s_boundary, g_boundary = ce.fit(conflicting_examples)

        # Version space must collapse because no hypothesis can simultaneously cover and exclude the same instance
        self.assertTrue(ce.is_version_space_empty)
        self.assertEqual(len(s_boundary), 0)
        self.assertEqual(len(g_boundary), 0)
        self.assertEqual(ce.predict(conflicting_examples[0]["features"]), "Undefined (Empty Version Space)")

    def test_helper_more_general(self):
        """Test generality partial order checks."""
        h_general = ("?", "?", "?", "?")
        h_specific = ("Clayey", "Tropical", "High", "Heavy")
        h_zero = ("0", "0", "0", "0")

        self.assertTrue(is_more_general_or_equal(h_general, h_specific))
        self.assertTrue(is_more_general_or_equal(h_specific, h_zero))
        self.assertFalse(is_more_general_or_equal(h_specific, h_general))


if __name__ == "__main__":
    unittest.main()
