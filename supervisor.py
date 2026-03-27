"""
Supervisor graph: routes onboarding flow by `stage` across five agent subgraphs.
"""

from __future__ import annotations

from typing import Any, Final

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agents.aml.aml_screening import build_aml_graph
from agents.doc_verification.doc_verify import build_doc_verify_graph
from agents.fraud_check.fraud_check import build_fraud_check_graph
from agents.doc_verification.kyc_approval import build_kyc_approval_graph
from agents.data_capture.data_capture import build_data_capture_graph
from agents.decision.decision_tool import build_decision_graph
from agents.decision.otp_verification import otp_verification_node
from state import OnboardingState

STAGE_TO_NODE: Final[dict[str, str]] = {
    "data_capture": "data_capture",
    "doc_verification": "doc_verification",
    "kyc_approval": "kyc_approval",
    "aml_screening": "aml_screening",
    "fraud_check": "fraud_check",
    "decision_agent": "decision_agent",
    "otp_verification": "otp_verification_handler",
}

SUBGRAPH_NODE_NAMES: Final[tuple[str, ...]] = tuple(set(STAGE_TO_NODE.values()))

PATH_MAP: dict[Any, str] = {
    "data_capture": "data_capture",
    "doc_verification": "doc_verification",
    "kyc_approval": "kyc_approval",
    "aml_screening": "aml_screening",
    "fraud_check": "fraud_check",
    "decision_agent": "decision_agent",
    "otp_verification_handler": "otp_verification_handler",
    END: END,
}


def route(state: OnboardingState, completed: str | None = None) -> Any:
    """
    Return the next supervisor node from `state["stage"]`.

    When `completed` is the name of the subgraph that just finished, and the
    next target for the current stage is the same subgraph, return END so the
    parent graph stops after one subgraph run per user turn (wait for the next
    HTTP invoke). When `completed` is None (routing from START), always enter
    the subgraph for the current stage.
    """
    stage = state["stage"]
    if stage in (
        "complete",
        "rejected",
        "otp_verification",
        "manual_review",
        "pending_docs",
        "escalated",
    ):
        return END
    target = STAGE_TO_NODE.get(stage)
    if target is None:
        return END
    if completed is not None and target == completed:
        return END
    return target


def build_supervisor() -> CompiledStateGraph:
    workflow = StateGraph(OnboardingState)

    workflow.add_node("data_capture", build_data_capture_graph())
    workflow.add_node("doc_verification", build_doc_verify_graph())
    workflow.add_node("kyc_approval", build_kyc_approval_graph())
    workflow.add_node("aml_screening", build_aml_graph())
    workflow.add_node("fraud_check", build_fraud_check_graph())
    workflow.add_node("decision_agent", build_decision_graph())
    workflow.add_node("otp_verification_handler", otp_verification_node)

    # Stage-driven entry (replaces a fixed set_entry_point("data_capture") so
    # later stages are reachable on each new invoke).
    workflow.add_conditional_edges(START, route, PATH_MAP)

    for name in SUBGRAPH_NODE_NAMES:
        workflow.add_conditional_edges(
            name,
            lambda s, n=name: route(s, n),
            PATH_MAP,
        )

    return workflow.compile()


onboarding_graph = build_supervisor()
