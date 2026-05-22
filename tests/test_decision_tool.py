"""Unit tests for deterministic decision routing."""

import sys
import unittest
from unittest.mock import MagicMock

sys.modules.setdefault("jwt", MagicMock())

from agents.decision.decision_tool import _FRAUD_BLOCKING, resolve_deterministic_decision


def _base_state(**overrides):
    state = {
        "session_id": "sess-1",
        "audit_session_id": "audit-1",
        "fraud_risk_score": 30,
        "fraud_status": "clear",
        "aml_status": "clear",
        "aml_completed": True,
        "aml_risk_score": 10,
        "kyc_data": {},
    }
    state.update(overrides)
    return state


class TestResolveDeterministicDecision(unittest.TestCase):
    def test_aml_flagged_escalates(self):
        result = resolve_deterministic_decision(
            _base_state(aml_status="flagged", fraud_status="clear")
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["action"], "escalate")

    def test_aml_review_queues_urgent(self):
        result = resolve_deterministic_decision(
            _base_state(aml_status="review", fraud_status="clear")
        )
        self.assertEqual(result["action"], "queue_for_review")
        self.assertEqual(result.get("priority"), "urgent")

    def test_fraud_review_blocks_approve(self):
        result = resolve_deterministic_decision(
            _base_state(fraud_status="review", fraud_risk_score=30)
        )
        self.assertIsNone(result)

    def test_pending_aml_review_blocks_approve(self):
        result = resolve_deterministic_decision(
            _base_state(fraud_status="pending_aml_review", fraud_risk_score=20)
        )
        self.assertIsNone(result)

    def test_approve_only_when_safe(self):
        result = resolve_deterministic_decision(_base_state())
        self.assertIsNotNone(result)
        self.assertEqual(result["action"], "approve")

    def test_approve_blocked_when_aml_incomplete(self):
        result = resolve_deterministic_decision(
            _base_state(aml_completed=False, aml_status="checking")
        )
        self.assertEqual(result["action"], "queue_for_review")

    def test_fraud_blocking_set_covers_review_states(self):
        self.assertIn("review", _FRAUD_BLOCKING)
        self.assertIn("pending_aml_review", _FRAUD_BLOCKING)


if __name__ == "__main__":
    unittest.main()
