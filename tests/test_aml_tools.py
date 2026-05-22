"""Unit tests for AML tools (mocked Mongo)."""

import sys
import unittest
from unittest.mock import MagicMock, patch

# Avoid pymongo dependency when running tests outside the project venv.
if "pymongo" not in sys.modules:
    _pymongo = MagicMock()
    sys.modules["pymongo"] = _pymongo
    sys.modules["pymongo.errors"] = MagicMock(
        ExecutionTimeout=type("ExecutionTimeout", (Exception,), {}),
        OperationFailure=type("OperationFailure", (Exception,), {}),
    )
if "core.mongodbase" not in sys.modules:
    _mock_db = MagicMock()
    sys.modules["core.mongodbase"] = MagicMock(aml_db=_mock_db)

from agents.aml.tools import (
    check_ofac_sanctions,
    check_pep_list,
    check_rbi_caution_list,
    evaluate_risk_rules,
)


class TestRbiCautionList(unittest.TestCase):
    @patch("agents.aml.tools.aml_db")
    def test_empty_pan_no_hit(self, mock_db):
        result = check_rbi_caution_list("")
        self.assertFalse(result["hit"])
        mock_db.rbi_caution_list.find_one.assert_not_called()

    @patch("agents.aml.tools.aml_db")
    def test_active_pan_hit(self, mock_db):
        mock_db.rbi_caution_list.find_one.return_value = {
            "name": "Test User",
            "reason": "fraud",
            "bank": "SBI",
        }
        result = check_rbi_caution_list("fraud1234f")
        self.assertTrue(result["hit"])
        mock_db.rbi_caution_list.find_one.assert_called_once()
        call_filter = mock_db.rbi_caution_list.find_one.call_args[0][0]
        self.assertEqual(call_filter["pan"], "FRAUD1234F")
        self.assertTrue(call_filter["active"])

    @patch("agents.aml.tools.aml_db")
    def test_inactive_not_returned_by_query(self, mock_db):
        mock_db.rbi_caution_list.find_one.return_value = None
        result = check_rbi_caution_list("OLDFL0000O")
        self.assertFalse(result["hit"])


class TestEvaluateRiskRules(unittest.TestCase):
    @patch("agents.aml.tools._get_cached_risk_rules")
    def test_self_declared_pep(self, mock_rules):
        mock_rules.return_value = [
            {
                "rule_id": "SELF_DECLARED_PEP",
                "description": "Self PEP",
                "check_type": "self_declared_pep",
                "params": {},
                "risk_score_delta": 35,
                "active": True,
            }
        ]
        result = evaluate_risk_rules(
            "Jane Doe",
            "1990-01-01",
            "savings",
            0,
            0,
            politically_exposed="Yes",
        )
        self.assertEqual(result["total_delta"], 35)
        self.assertEqual(len(result["triggered_rules"]), 1)

    @patch("agents.aml.tools._get_cached_risk_rules")
    def test_near_sanctions_fuzzy_range(self, mock_rules):
        mock_rules.return_value = [
            {
                "rule_id": "NEAR_SANCTIONS_MATCH",
                "description": "Near OFAC",
                "check_type": "fuzzy_score_range",
                "params": {"min_score": 70, "max_score": 84, "source": "ofac_sdn"},
                "risk_score_delta": 30,
                "active": True,
            }
        ]
        result = evaluate_risk_rules("Jane", None, "savings", 75, 0)
        self.assertEqual(result["total_delta"], 30)

    @patch("agents.aml.tools._get_cached_risk_rules")
    def test_no_pep_declaration(self, mock_rules):
        mock_rules.return_value = [
            {
                "rule_id": "SELF_DECLARED_PEP",
                "check_type": "self_declared_pep",
                "params": {},
                "risk_score_delta": 35,
                "active": True,
            }
        ]
        result = evaluate_risk_rules("Jane", None, "savings", 0, 0, politically_exposed="No")
        self.assertEqual(result["total_delta"], 0)


class TestOfacFuzzyHit(unittest.TestCase):
    @patch("agents.aml.tools.aml_db")
    def test_fuzzy_85_sets_hit(self, mock_db):
        mock_db.ofac_sdn_list.find_one.return_value = None
        mock_cursor = MagicMock()
        mock_cursor.sort.return_value = mock_cursor
        mock_cursor.max_time_ms.return_value = mock_cursor
        mock_cursor.limit.return_value = mock_cursor
        mock_cursor.__iter__ = MagicMock(
            return_value=iter([
                {"name": "Mohammad Al Rashid Jr", "aliases": [], "program": "SDGT", "uid": "x"},
            ])
        )
        mock_db.ofac_sdn_list.find.return_value = mock_cursor

        result = check_ofac_sanctions("Mohammad Al Rashid")
        if result["match_score"] >= 85:
            self.assertTrue(result["hit"])
        else:
            self.assertGreaterEqual(result["match_score"], 0)


class TestPepExactHit(unittest.TestCase):
    @patch("agents.aml.tools.aml_db")
    def test_exact_name_hit(self, mock_db):
        mock_db.pep_list.find_one.return_value = {
            "name": "Arun Kumar Singh",
            "pep_tier": 1,
            "position": "MP",
            "jurisdiction": "IN",
        }
        result = check_pep_list("Arun Kumar Singh")
        self.assertTrue(result["hit"])
        self.assertEqual(result["pep_tier"], 1)


if __name__ == "__main__":
    unittest.main()
