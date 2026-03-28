"""
Decision Making Agent — LangGraph subgraph.

Receives fraud and AML signals and decides the next step for each account
application using one of five tools:
  approve_account, queue_for_review, reject_application,
  request_additional_docs, escalate_to_compliance.

Usage:
    from agents.decision.decision_tool import build_decision_graph
    graph = build_decision_graph()
    result = graph.invoke(state)
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from audit_logger import AuditLogger
from llm_config import AgentLLM
from memory_manager import AgentMemoryManager
from agents.decision.activation_service import send_activation_email
from state import OnboardingState

logger = logging.getLogger(__name__)

# Shared instances
_audit = AuditLogger()
_memory = AgentMemoryManager()

# ---------------------------------------------------------------------------
# SECTION 1 — Decision Tools
# ---------------------------------------------------------------------------


@tool
def approve_account(session_id: str, reason: str) -> dict:
    """Approve the account application. Use when all checks pass and risk is low."""
    return {
        "action": "approve",
        "stage": "complete",
        "progress": 100,
        "decision_reason": reason,
        "user_message": (
            "Congratulations! Your AccuEntry account has been "
            "approved and is ready to use. Welcome aboard."
        ),
    }


@tool
def queue_for_review(
    session_id: str, reason: str, priority: str = "normal"
) -> dict:
    """Queue the application for manual review. Use when risk is ambiguous."""
    return {
        "action": "queue_for_review",
        "stage": "manual_review",
        "progress": 85,
        "decision_reason": reason,
        "priority": priority,
        "user_message": (
            "Your application is under additional review by our "
            "compliance team. We will contact you within "
            "1–2 business days."
        ),
    }


@tool
def reject_application(session_id: str, reason: str) -> dict:
    """Reject the application. Use when fraud is confirmed or AML flagged."""
    return {
        "action": "reject",
        "stage": "rejected",
        "progress": 100,
        "decision_reason": reason,
        "user_message": (
            "We are unable to proceed with your application at "
            "this time. If you believe this is an error, please "
            "contact our support team with reference ID: "
            f"{session_id[:8].upper()}"
        ),
    }


@tool
def request_additional_docs(
    session_id: str, reason: str, doc_list: list[str]
) -> dict:
    """Request additional documents from the applicant."""
    return {
        "action": "request_docs",
        "stage": "pending_docs",
        "progress": 60,
        "decision_reason": reason,
        "pending_docs": doc_list,
        "user_message": (
            "We need a few more documents to complete your "
            f"application: {', '.join(doc_list)}. Please "
            "re-upload these to continue."
        ),
    }


@tool
def escalate_to_compliance(
    session_id: str, reason: str, severity: str = "high"
) -> dict:
    """Escalate to senior compliance officer. Use for PEP, sanctions, or critical AML."""
    return {
        "action": "escalate",
        "stage": "escalated",
        "progress": 90,
        "decision_reason": reason,
        "severity": severity,
        "user_message": (
            "Your application requires senior compliance review. "
            "A dedicated officer will contact you within 24 hours."
        ),
    }


ALL_DECISION_TOOLS = [
    approve_account,
    queue_for_review,
    reject_application,
    request_additional_docs,
    escalate_to_compliance,
]

_TOOL_MAP: dict[str, Any] = {t.name: t for t in ALL_DECISION_TOOLS}

# ---------------------------------------------------------------------------
# SECTION 2 — System Prompt
# ---------------------------------------------------------------------------

DECISION_AGENT_SYSTEM_PROMPT = """
You are a senior compliance officer at AccuEntry, an RBI-regulated digital
bank. You receive the output of automated fraud and AML checks and decide
the next step for each account application.

You have exactly five tools available:
  approve_account          — use when all checks pass and risk is low
  queue_for_review         — use when risk is ambiguous or one signal
                             is elevated but not conclusive
  reject_application       — use when fraud is confirmed or AML flagged
  request_additional_docs  — use when a specific document failed
                             verification but the applicant seems genuine
  escalate_to_compliance   — use when a PEP match, sanctions hit, or
                             critical AML flag is present

For every tool call, set argument session_id to the audit_session_id value
from the Application Risk Summary (same string).

Decision rules (follow strictly):
  1. If aml_status = "flagged"
     → ALWAYS escalate_to_compliance, never approve

  2. If fraud_status = "flagged" AND fraud_score >= 60
     → reject_application

  3. If fraud_score <= 59 AND aml_status = "clear"
     → approve_account

  4. If fraud_score <= 59 AND aml_status in
     ("checking", "pending", "pending_aml", null, None)
     → approve_account
     (AML still processing but fraud risk is low — safe to proceed)

  5. If fraud_score >= 60 AND fraud_score < 80
     → queue_for_review with priority "normal"

  6. If fraud_score >= 80
     → reject_application

  7. If source_of_funds = "cryptocurrency" or "cash"
     AND annual_income > 1000000
     → queue_for_review with priority "urgent"

  8. If any field is null, missing, or not yet collected
     → assume lowest risk for that field only
     → do NOT escalate, reject, or request docs
        due to a missing field alone
     → request_additional_docs ONLY when a specific document
        explicitly failed verification (OCR failure, photo
        mismatch, PAN invalid) — never because a field is null

You must call exactly ONE tool per decision. After calling the tool,
respond ONLY with the JSON output of the tool — no additional text.
""".strip()

# ---------------------------------------------------------------------------
# SECTION 3 — Context Builder
# ---------------------------------------------------------------------------


def _build_decision_context(state: OnboardingState) -> str:
    fraud_score = state.get("fraud_risk_score") or 0
    kyc_data: dict = state.get("kyc_data") or {}

    context: dict[str, Any] = {
        "fraud_status": state.get("fraud_status"),
        "fraud_score": fraud_score,
        "fraud_flags": (state.get("fraud_signals") or [])[:5],
        "aml_status": state.get("aml_status"),
        # "risk_model_label": state.get("risk_model_label"),  # TODO: not yet implemented — risk_analysis
        # "risk_model_confidence": state.get("risk_model_confidence"),  # TODO: not yet implemented — risk_analysis
        # "video_kyc_status": state.get("video_kyc_status"),  # TODO: not yet implemented — video_kyc
        "source_of_funds": kyc_data.get("source_of_funds") or state.get("source_of_funds"),
        "annual_income": kyc_data.get("annual_income") or state.get("annual_income"),
        "nationality": kyc_data.get("nationality") or state.get("nationality"),
        "audit_session_id": state.get("audit_session_id"),
    }

    similar = _memory.retrieve_similar(
        "decision_agent", query_data=context, top_k=3
    )

    lines = ["=== Application Risk Summary ==="]
    for key, val in context.items():
        lines.append(f"  {key}: {val}")

    if similar:
        lines.append("\nSimilar past cases:")
        for i, case in enumerate(similar, 1):
            out = case.get("output_data", {})
            lines.append(
                f"  Case {i}: decision={out.get('action', 'N/A')}, "
                f"risk_score={case.get('risk_score', 'N/A')}"
            )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# SECTION 4 — Agentic Node
# ---------------------------------------------------------------------------


async def _run_decision_agent(state: OnboardingState) -> dict[str, Any]:
    session_id = state.get("audit_session_id", "unknown")
    fraud_score = state.get("fraud_risk_score") or 0

    logger.info(
        "decision_agent_state_check | session=%s | fraud_status=%s "
        "fraud_score=%s | aml_status=%s | pan=%s | aadhaar=%s | face=%s",
        session_id,
        state.get("fraud_status"),
        state.get("fraud_risk_score"),
        state.get("aml_status"),
        state.get("pan_verified"),
        state.get("aadhaar_verified"),
        state.get("face_verified"),
    )

    input_snapshot = {
        "fraud_score": fraud_score,
        "fraud_status": state.get("fraud_status"),
        "aml_status": state.get("aml_status"),
        "aml_risk_score": state.get("aml_risk_score"),
        "pan_verified": state.get("pan_verified"),
        "aadhaar_verified": state.get("aadhaar_verified"),
        "face_verified": state.get("face_verified"),
        # "video_kyc_status": state.get("video_kyc_status"),  # TODO: not yet implemented — video_kyc
    }

    logger.info(
        "decision_agent_start: all_tool_outputs=%s",
        input_snapshot,
    )

    # 1. Log entry
    _audit.log_event(
        session_id,
        "decision_agent_start",
        input_data=input_snapshot,
    )

    # 2. Build context
    context_string = _build_decision_context(state)

    # 3. Bind tools to LLM
    llm = AgentLLM().get_llm("decision_agent")
    llm_with_tools = llm.bind_tools(ALL_DECISION_TOOLS, tool_choice="required")

    # 4. Agentic loop (max 3 iterations)
    messages: list = [
        SystemMessage(content=DECISION_AGENT_SYSTEM_PROMPT),
        HumanMessage(content=context_string),
    ]

    tool_result: dict[str, Any] | None = None

    for _ in range(3):
        response = await llm_with_tools.ainvoke(messages)
        messages.append(response)

        if not response.tool_calls:
            break

        tc = response.tool_calls[0]
        result = _TOOL_MAP[tc["name"]].invoke(tc["args"])
        messages.append(
            ToolMessage(
                content=json.dumps(result),
                tool_call_id=tc["id"],
            )
        )
        tool_result = result
        break  # one tool call per decision

        # 5. Build state update from tool result
    if tool_result is None:
        tool_result = queue_for_review.invoke({
            "session_id": session_id,
            "reason": "LLM did not invoke any tool — routed to manual review",
            "priority": "normal",
        })

    if tool_result.get("action") == "approve":
        from agents.decision.otp_service import generate_otp, send_otp_email, mask_email

        email_id = state.get("email_id") or ""
        full_name = state.get("full_name") or "User"

        otp_code = generate_otp(session_id)
        if otp_code is None:
            tool_result["user_message"] = (
                "We're having trouble sending your activation code. "
                "Please try again later or contact support."
            )
        else:
            email_sent = await send_otp_email(session_id, email_id, otp_code)
            masked = mask_email(email_id)

            if email_sent:
                import json as _json
                otp_msg = {
                    "type": "OTP_REQUESTED",
                    "channel": "chatbot",
                    "payload": {
                        "message": (
                            "We've reached the final stage of setting up your account!\n"
                            f"We have sent a 4-digit Activation Code to your email {masked}.\n"
                            "Please enter the code to activate your account."
                        ),
                        "inputType": "otp",
                        "otpLength": 4,
                        "expiresInMinutes": 10,
                    },
                }
                tool_result["user_message"] = _json.dumps(otp_msg)
                tool_result["stage"] = "otp_verification"
            else:
                tool_result["user_message"] = (
                    "We're having trouble sending your activation code. "
                    "Please try again or contact support."
                )

    update: dict[str, Any] = {
        "stage": tool_result["stage"],
        "decision_reason": tool_result["decision_reason"],
        "decision_action": tool_result["action"],
        "progress": tool_result["progress"],
        "admin_override": False,
        "messages": [
            {"role": "assistant", "text": tool_result["user_message"]}
        ],
    }

    if "pending_docs" in tool_result:
        update["pending_docs"] = tool_result["pending_docs"]

    # 6. Store interaction in memory
    _memory.store_interaction(
        session_id=session_id,
        agent_name="decision_agent",
        input_data=input_snapshot,
        output_data={
            "action": tool_result["action"],
            "reason": tool_result["decision_reason"],
        },
        risk_score=fraud_score,
        decision=tool_result["action"],
    )

    # 7. Log exit
    _audit.log_event(
        session_id,
        "decision_agent_complete",
        output_data={
            "action": tool_result["action"],
            "stage": tool_result["stage"],
            "reason": tool_result["decision_reason"],
        },
        decision=tool_result["action"],
    )

    return update


async def decision_agent_node(state: OnboardingState) -> dict[str, Any]:
    """Async decision node with 15-second timeout. Falls back to manual_review."""
    session_id = state.get("audit_session_id", "unknown")
    try:
        return await asyncio.wait_for(_run_decision_agent(state), timeout=15.0)
    except asyncio.TimeoutError:
        logger.warning("Decision agent timed out for session %s", session_id)
        _audit.log_event(
            session_id,
            "decision_agent_error",
            metadata={"error": "Timeout after 15 seconds"},
        )
        return {
            "stage": "manual_review",
            "decision_reason": "Decision agent timed out — routed to manual review",
            "decision_action": "queue_for_review",
            "progress": 85,
            "admin_override": False,
            "messages": [
                {
                    "role": "assistant",
                    "text": (
                        "Your application is under additional review by our "
                        "compliance team. We will contact you within "
                        "1–2 business days."
                    ),
                }
            ],
        }
    except Exception as exc:
        logger.exception("Decision agent error for session %s", session_id)
        _audit.log_event(
            session_id,
            "decision_agent_error",
            metadata={"error": str(exc)},
        )
        return {
            "stage": "manual_review",
            "decision_reason": "Decision agent error — routed to manual review",
            "decision_action": "queue_for_review",
            "progress": 85,
            "admin_override": False,
            "messages": [
                {
                    "role": "assistant",
                    "text": (
                        "Your application is under additional review by our "
                        "compliance team. We will contact you within "
                        "1–2 business days."
                    ),
                }
            ],
        }


# ---------------------------------------------------------------------------
# SECTION 5 — Routing & Notification Stubs
# ---------------------------------------------------------------------------


def route_decision(state: OnboardingState) -> str:
    stage = state.get("stage", "")
    mapping: dict[str, str] = {
        "otp_verification": END,
        "complete": END,
        "manual_review": "notify_review_queue",
        "rejected": "notify_rejection",
        "pending_docs": "notify_pending_docs",
        "escalated": "notify_escalation",
    }
    return mapping.get(stage, END)


def notify_review_queue(state: OnboardingState) -> dict:
    _audit.log_event(
        state.get("audit_session_id", "unknown"),
        "queued_for_review",
        metadata={
            "decision_reason": state.get("decision_reason"),
        },
    )
    # Empty merge: terminal stage and messages already set by decision_agent.
    return {}


def notify_rejection(state: OnboardingState) -> dict:
    _audit.log_event(
        state.get("audit_session_id", "unknown"),
        "application_rejected",
        metadata={
            "decision_reason": state.get("decision_reason"),
        },
    )
    # Empty merge: terminal stage and messages already set by decision_agent.
    return {}


def notify_pending_docs(state: OnboardingState) -> dict:
    _audit.log_event(
        state.get("audit_session_id", "unknown"),
        "pending_docs_requested",
        metadata={
            "pending_docs": state.get("pending_docs", []),
            "decision_reason": state.get("decision_reason"),
        },
    )
    # Empty merge: terminal stage and messages already set by decision_agent.
    return {}


def notify_escalation(state: OnboardingState) -> dict:
    _audit.log_event(
        state.get("audit_session_id", "unknown"),
        "escalated_to_compliance",
        metadata={
            "decision_reason": state.get("decision_reason"),
        },
    )
    # Empty merge: terminal stage and messages already set by decision_agent.
    return {}


# ---------------------------------------------------------------------------
# SECTION 6 — Graph Assembly
# ---------------------------------------------------------------------------


def build_decision_graph() -> CompiledStateGraph:
    workflow = StateGraph(OnboardingState)

    workflow.add_node("decision_agent", decision_agent_node)
    workflow.add_node("notify_review_queue", notify_review_queue)
    workflow.add_node("notify_rejection", notify_rejection)
    workflow.add_node("notify_pending_docs", notify_pending_docs)
    workflow.add_node("notify_escalation", notify_escalation)

    workflow.add_edge(START, "decision_agent")
    workflow.add_conditional_edges(
        "decision_agent",
        route_decision,
        {
            END: END,
            "notify_review_queue": "notify_review_queue",
            "notify_rejection": "notify_rejection",
            "notify_pending_docs": "notify_pending_docs",
            "notify_escalation": "notify_escalation",
        },
    )

    workflow.add_edge("notify_review_queue", END)
    workflow.add_edge("notify_rejection", END)
    workflow.add_edge("notify_pending_docs", END)
    workflow.add_edge("notify_escalation", END)

    return workflow.compile()
