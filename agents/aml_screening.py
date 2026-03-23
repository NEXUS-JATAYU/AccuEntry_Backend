"""
AML screening subgraph (stub): always clears and advances to fraud check.

TODO: Integrate a real AML provider API (e.g. OFAC sanctions list, ComplyAdvantage)
to screen the customer before marking aml_status.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from state import OnboardingState


def screen_node(state: OnboardingState) -> dict[str, Any]:
    _ = state
    return {
        "aml_status": "clear",
        "stage": "fraud_check",
        "progress": 80,
        "messages": [
            {
                "role": "assistant",
                "text": (
                    "AML screening passed. Proceeding to final fraud check."
                ),
            }
        ],
    }


def build_aml_screening_graph() -> CompiledStateGraph:
    workflow = StateGraph(OnboardingState)
    workflow.add_node("screen_node", screen_node)
    workflow.add_edge(START, "screen_node")
    workflow.add_edge("screen_node", END)
    return workflow.compile()
