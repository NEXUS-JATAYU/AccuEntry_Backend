"""Unit tests for AML flag user report formatting."""

import unittest

from agents.aml.aml_user_report import (
    build_aml_flag_report_deterministic,
    extract_aml_flag_details,
)


class TestAmlUserReport(unittest.TestCase):
    def test_extract_ofac_hit(self):
        state = {
            "audit_session_id": "abcd-1234-efgh",
            "aml_risk_score": 90,
            "aml_raw_results": {
                "rbi": {"hit": False},
                "ofac": {"hit": True, "matched_name": "Test User", "match_score": 92},
                "pep": {"hit": False},
                "rules": {"triggered_rules": [{"rule_id": "HIGH_RISK", "description": "Elevated"}]},
            },
        }
        details = extract_aml_flag_details(state)
        self.assertEqual(len(details["flagged_checks"]), 1)
        self.assertIn("OFAC", details["flagged_checks"][0]["name"])

    def test_deterministic_report_includes_branch_and_reference(self):
        state = {
            "session_id": "sess-1",
            "aml_risk_score": 80,
            "aml_raw_results": {
                "rbi": {"hit": True, "reason": "PAN on caution list"},
                "ofac": {"hit": False},
                "pep": {"hit": False},
                "rules": {},
            },
        }
        details = extract_aml_flag_details(state)
        report = build_aml_flag_report_deterministic(details)
        self.assertIn("AML SCREENING REPORT", report)
        self.assertIn("nearest AccuEntry branch", report)
        self.assertIn("RBI Caution", report)


if __name__ == "__main__":
    unittest.main()
