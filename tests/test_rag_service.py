"""Unit tests for RAG query helpers."""

import unittest

from rag_service import build_faq_retrieval_query


class TestBuildFaqRetrievalQuery(unittest.TestCase):
    def test_complete_stage_adds_activation_hints(self):
        q = build_faq_retrieval_query("how long does review take", stage="complete")
        self.assertIn("post activation", q.lower())
        self.assertIn("how long does review take", q)

    def test_rejected_stage_adds_rejection_hints(self):
        q = build_faq_retrieval_query("why was I declined", stage="rejected")
        self.assertIn("rejected", q.lower())

    def test_includes_decision_action(self):
        q = build_faq_retrieval_query("next steps", stage="manual_review", decision_action="queue_for_review")
        self.assertIn("queue_for_review", q)


if __name__ == "__main__":
    unittest.main()
