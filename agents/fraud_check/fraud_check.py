"""
Fraud check subgraph (stub): always clears and completes onboarding.

TODO: Replace with real velocity / device fingerprinting / risk scoring before
approving the account.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from state import OnboardingState


def check_node(state: OnboardingState) -> dict[str, Any]:
    aml_status = state.get("aml_status")
    aml_in_background = bool(state.get("aml_in_background"))

    if aml_in_background or aml_status in (None, "pending", "checking"):
        return {
            "stage": "fraud_check",
            "progress": max(state.get("progress", 0), 80),
            "messages": [
                {
                    "role": "assistant",
                    "text": (
                        "AML screening is in progress. We will continue to final activation "
                        "as soon as compliance checks finish."
                    ),
                }
            ],
        }

    if aml_status == "flagged":
        return {
            "stage": "rejected",
            "fraud_status": "flagged",
            "messages": [
                {
                    "role": "assistant",
                    "text": (
                        "We are unable to proceed with this application after compliance review."
                    ),
                }
            ],
        }

    return {
        "fraud_status": "clear",
        "stage": "complete",
        "progress": 100,
        "messages": [
            {
                "role": "assistant",
                "text": (
                    "All checks complete. Your account has been approved! "
                    "Welcome to AccuEntry."
                ),
            }
        ],
    }


def build_fraud_check_graph() -> CompiledStateGraph:
    workflow = StateGraph(OnboardingState)
    workflow.add_node("check_node", check_node)
    workflow.add_edge(START, "check_node")
    workflow.add_edge("check_node", END)
    return workflow.compile()
