"""Unit tests for AML scoring and routing."""

import unittest

from agents.aml.aml_scoring import (
    AUTO_CLEAR_THRESHOLD,
    AUTO_FLAG_THRESHOLD,
    compute_risk_score,
    has_screening_failure,
    route_after_aggregate,
    route_by_score,
)


class TestComputeRiskScore(unittest.TestCase):
    def test_empty_results(self):
        self.assertEqual(compute_risk_score({}), 0)
        self.assertEqual(compute_risk_score(None), 0)

    def test_rbi_hit_flags_band(self):
        raw = {
            "rbi": {"hit": True},
            "ofac": {"hit": False},
            "pep": {"hit": False},
            "rules": {"total_delta": 0},
        }
        self.assertEqual(compute_risk_score(raw), 80)
        self.assertEqual(route_by_score(80), "auto_flag")

    def test_ofac_hit(self):
        raw = {
            "rbi": {"hit": False},
            "ofac": {"hit": True},
            "pep": {"hit": False},
            "rules": {"total_delta": 0},
        }
        self.assertEqual(compute_risk_score(raw), 90)

    def test_pep_tier_and_rules(self):
        raw = {
            "rbi": {"hit": False},
            "ofac": {"hit": False},
            "pep": {"hit": True, "pep_tier": 2},
            "rules": {"total_delta": 15},
        }
        self.assertEqual(compute_risk_score(raw), 35)
        self.assertEqual(route_by_score(35), "manual_review")

    def test_near_miss_rule_reaches_review(self):
        raw = {
            "rbi": {"hit": False},
            "ofac": {"hit": False, "match_score": 75},
            "pep": {"hit": False},
            "rules": {"total_delta": 30},
        }
        score = compute_risk_score(raw)
        self.assertEqual(score, 30)
        self.assertGreaterEqual(score, AUTO_CLEAR_THRESHOLD)
        self.assertLess(score, AUTO_FLAG_THRESHOLD)

    def test_capped_at_100(self):
        raw = {
            "rbi": {"hit": True},
            "ofac": {"hit": True},
            "pep": {"hit": True, "pep_tier": 1},
            "rules": {"total_delta": 60},
        }
        self.assertEqual(compute_risk_score(raw), 100)


class TestScreeningFailure(unittest.TestCase):
    def test_timeout_routes_manual_review(self):
        raw = {
            "rbi": {"hit": False, "error": "timeout"},
            "ofac": {"hit": False},
            "pep": {"hit": False},
            "rules": {"total_delta": 0},
        }
        self.assertTrue(has_screening_failure(raw))
        state = {"aml_raw_results": raw, "aml_risk_score": 0}
        self.assertEqual(route_after_aggregate(state), "manual_review")

    def test_clear_low_score(self):
        raw = {
            "rbi": {"hit": False},
            "ofac": {"hit": False, "match_score": 0},
            "pep": {"hit": False},
            "rules": {"total_delta": 0},
        }
        state = {"aml_raw_results": raw, "aml_risk_score": compute_risk_score(raw)}
        self.assertEqual(route_after_aggregate(state), "auto_clear")


if __name__ == "__main__":
    unittest.main()
