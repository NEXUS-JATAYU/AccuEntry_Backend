"""Unit tests for fraud check scoring and routing."""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Optional runtime deps may be absent in minimal test environments
sys.modules.setdefault("phonenumbers", MagicMock())
sys.modules.setdefault("email_validator", MagicMock())

from agents.fraud_check.fraud_check import (
    layer3_identity,
    resolve_fraud_action,
)


class TestResolveFraudAction(unittest.TestCase):
    def test_score_55_llm_manual_review_not_cleared(self):
        action, score = resolve_fraud_action(55, "manual_review", skip_llm_below=25)
        self.assertEqual(action, "manual_review")
        self.assertEqual(score, 55)

    def test_score_55_llm_clear_becomes_manual_review(self):
        action, _ = resolve_fraud_action(55, "clear", skip_llm_below=25)
        self.assertEqual(action, "manual_review")

    def test_score_60_never_auto_clear(self):
        action, _ = resolve_fraud_action(60, "clear", skip_llm_below=25)
        self.assertEqual(action, "manual_review")

    def test_score_75_clear_becomes_reject(self):
        action, _ = resolve_fraud_action(75, "clear", skip_llm_below=25)
        self.assertEqual(action, "reject")

    def test_below_skip_llm_threshold_always_clear(self):
        action, score = resolve_fraud_action(20, "reject", skip_llm_below=25)
        self.assertEqual(action, "clear")
        self.assertEqual(score, 20)

    @patch.dict(os.environ, {"FRAUD_FORCE_CLEAR_BELOW": "60"}, clear=False)
    def test_legacy_force_clear_env(self):
        action, score = resolve_fraud_action(55, "manual_review", skip_llm_below=25)
        self.assertEqual(action, "clear")
        self.assertEqual(score, 55)


class TestLayer3Identity(unittest.TestCase):
    def test_dob_mismatch_adds_signal(self):
        state = {
            "full_name": "Jane Doe",
            "dob": "1990-01-15",
            "document_dob": "1985-06-20",
            "document_name": "Jane Doe",
        }
        score, signals = layer3_identity(state)
        self.assertGreater(score, 0)
        self.assertTrue(any("dob_mismatch" in s for s in signals))

    def test_uses_dob_not_date_of_birth_only(self):
        state = {
            "dob": "1990-01-15",
            "document_dob": "1990-01-15",
            "document_name": "Test User",
            "full_name": "Test User",
        }
        score, signals = layer3_identity(state)
        self.assertEqual(score, 0)
        self.assertFalse(any("dob_mismatch" in s for s in signals))


if __name__ == "__main__":
    unittest.main()
