"""
Supervisor graph: routes onboarding flow by `stage` across five agent subgraphs.
"""

from __future__ import annotations

import logging
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

logger = logging.getLogger(__name__)

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
    sid = state.get("session_id")
    audit = state.get("audit_session_id")
    print(f"[DEBUG][supervisor] sid={sid} audit={audit} stage={stage} completed={completed}")
    if stage in (
        "complete",
        "rejected",
        "manual_review",
        "pending_docs",
        "escalated",
    ):
        logger.info("supervisor_route | stage=%s → node=%s", stage, END)
        return END

    # Sequential guard: decisioning cannot run until AML and fraud are both clear.
    if stage == "decision_agent":
        aml_status = (state.get("aml_status") or "").lower()
        fraud_status = (state.get("fraud_status") or "").lower()
        if aml_status != "clear" or fraud_status != "clear":
            logger.info(
                "supervisor_route | blocked decision_agent aml=%s fraud=%s → node=%s",
                aml_status,
                fraud_status,
                "fraud_check",
            )
            return "fraud_check"

    target = STAGE_TO_NODE.get(stage)
    if target is None:
        logger.info("supervisor_route | stage=%s → node=%s", stage, END)
        return END
    # Critical guard: do not execute OTP handler in the same run that produced
    # the OTP prompt. Only route to OTP handler on the next /chat turn when
    # routing from START (completed is None).
    if stage == "otp_verification" and completed is not None:
        logger.info("supervisor_route | stage=%s completed=%s → node=%s", stage, completed, END)
        return END
    if completed is not None and target == completed:
        logger.info("supervisor_route | stage=%s → node=%s", stage, END)
        return END
    logger.info("supervisor_route | stage=%s → node=%s", stage, target)
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
