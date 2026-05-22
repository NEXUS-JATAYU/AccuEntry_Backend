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

import os
from rag_service import retrieve_as_context
from audit_logger import AuditLogger
from llm_config import AgentLLM
from memory_manager import AgentMemoryManager
from agents.decision.activation_service import send_activation_email
from agents.faq.faq_agent import POST_PROCESS_FAQ_INVITE
from state import OnboardingState

logger = logging.getLogger(__name__)

# Fraud statuses that must never receive account approval
_FRAUD_BLOCKING = frozenset({
    "flagged",
    "rejected",
    "review",
    "pending_aml",
    "pending_aml_review",
})

# Terminal stages where users may ask process questions (RAG FAQ in /chat).
_FAQ_INVITE_STAGES = frozenset({
    "manual_review",
    "rejected",
    "escalated",
    "pending_docs",
    "complete",
})


def _append_faq_invite(messages: list[dict]) -> list[dict]:
    """Append post-process FAQ invite to the last plain-text assistant message."""
    if not messages:
        return [{"role": "assistant", "text": POST_PROCESS_FAQ_INVITE}]
    updated = list(messages)
    last = updated[-1]
    if not isinstance(last, dict) or last.get("role") != "assistant":
        updated.append({"role": "assistant", "text": POST_PROCESS_FAQ_INVITE})
        return updated
    if last.get("type") in {"OTP_REQUESTED", "ACCOUNT_ACTIVATED"} or isinstance(
        last.get("payload"), dict
    ):
        updated.append({"role": "assistant", "text": POST_PROCESS_FAQ_INVITE})
        return updated
    text = str(last.get("text") or "").strip()
    if POST_PROCESS_FAQ_INVITE not in text:
        updated[-1] = {**last, "text": f"{text}\n\n{POST_PROCESS_FAQ_INVITE}".strip()}
    return updated

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
     AND aml_completed = true
     AND fraud_status NOT in ("flagged", "rejected", "review",
         "pending_aml", "pending_aml_review")
     → approve_account

  4. If aml_status in ("checking", "pending", "pending_aml", null, None)
     OR aml_completed = false
     → queue_for_review with priority "normal"
     (Never approve while AML is incomplete)

  5. If aml_status = "review"
     → queue_for_review with priority "urgent"

  6. If fraud_score >= 60 AND fraud_score < 80
     → queue_for_review with priority "normal"

  7. If fraud_score >= 80
     → reject_application

  8. If source_of_funds = "cryptocurrency" or "cash"
     AND annual_income > 1000000
     → queue_for_review with priority "urgent"

  9. If any field is null, missing, or not yet collected
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


def _decision_rag_query(state: OnboardingState, fraud_score: int, aml_status: str | None) -> str:
    """Risk-aware retrieval query for decision LLM policy context."""
    aml = (aml_status or "").lower()
    if aml == "flagged":
        return "AML flagged escalation compliance sanctions"
    if aml == "review":
        return "AML manual review compliance timeline urgent queue"
    if aml in {"pending", "checking"} or not state.get("aml_completed"):
        return "AML screening in progress pending completion"
    if fraud_score >= 80:
        return "fraud rejection criteria critical risk score"
    if fraud_score >= 60:
        return "fraud manual review elevated risk score queue"
    return "account approval criteria low risk KYC AML clear"


def _aml_triggered_rules(state: OnboardingState, limit: int = 5) -> list[str]:
    raw = state.get("aml_raw_results") or {}
    rules = raw.get("rules") if isinstance(raw, dict) else None
    if not isinstance(rules, dict):
        return []
    triggered = rules.get("triggered") or rules.get("triggered_rules") or []
    if isinstance(triggered, list):
        return [str(r) for r in triggered[:limit]]
    return []


def _build_decision_context(state: OnboardingState) -> str:
    fraud_score = state.get("fraud_risk_score") or 0
    kyc_data: dict = state.get("kyc_data") or {}

    context: dict[str, Any] = {
        "fraud_status": state.get("fraud_status"),
        "fraud_score": fraud_score,
        "fraud_flags": (state.get("fraud_signals") or [])[:5],
        "aml_status": state.get("aml_status"),
        "aml_risk_score": state.get("aml_risk_score"),
        "aml_completed": state.get("aml_completed"),
        "aml_triggered_rules": _aml_triggered_rules(state),
        # "risk_model_label": state.get("risk_model_label"),  # TODO: not yet implemented — risk_analysis
        # "risk_model_confidence": state.get("risk_model_confidence"),  # TODO: not yet implemented — risk_analysis
        # "video_kyc_status": state.get("video_kyc_status"),  # TODO: not yet implemented — video_kyc
        "source_of_funds": kyc_data.get("source_of_funds") or state.get("source_of_funds"),
        "annual_income": kyc_data.get("annual_income") or state.get("annual_income"),
        "nationality": kyc_data.get("nationality") or state.get("nationality"),
        "audit_session_id": state.get("audit_session_id"),
    }

    use_rag = os.getenv("DECISION_USE_RAG", "true").lower() in {"1", "true", "yes"}
    policy_excerpt = (
        retrieve_as_context(
            _decision_rag_query(state, fraud_score, context.get("aml_status")),
            top_k=3,
        )
        if use_rag
        else ""
    )

    similar = _memory.retrieve_similar(
        "decision_agent",
        query_data=context,
        top_k=3,
        where={"event_type": "decision_outcome"},
    )

    lines = ["=== Application Risk Summary ==="]
    for key, val in context.items():
        lines.append(f"  {key}: {val}")

    if policy_excerpt:
        lines.append("\nRelevant policy excerpt:")
        lines.append(policy_excerpt)

    if similar:
        lines.append("\nSimilar past cases:")
        for i, case in enumerate(similar, 1):
            out = case.get("output_data", {})
            lines.append(
                f"  Case {i}: decision={out.get('action', 'N/A')}, "
                f"risk_score={case.get('risk_score', 'N/A')}"
            )

    return "\n".join(lines)


def _is_low_risk_approval_candidate(state: OnboardingState) -> bool:
    aml_status = (state.get("aml_status") or "").lower()
    fraud_status = (state.get("fraud_status") or "").lower()
    fraud_score = state.get("fraud_risk_score") or 0
    return (
        aml_status == "clear"
        and state.get("aml_completed")
        and fraud_status not in _FRAUD_BLOCKING
        and fraud_score <= 59
    )


def resolve_deterministic_decision(state: OnboardingState) -> dict[str, Any] | None:
    """
    Apply deterministic decision rules (same precedence as _run_decision_agent).
    Returns tool result dict or None when LLM path should run.
    """
    session_id = state.get("audit_session_id") or state.get("session_id") or "unknown"
    fraud_score = state.get("fraud_risk_score") or 0
    aml_status = state.get("aml_status")
    fraud_status = (state.get("fraud_status") or "").lower()
    kyc_data: dict[str, Any] = state.get("kyc_data") or {}
    source_of_funds = (
        kyc_data.get("source_of_funds") or state.get("source_of_funds") or ""
    ).strip().lower()
    annual_income_raw = kyc_data.get("annual_income") or state.get("annual_income")
    try:
        annual_income = (
            float(annual_income_raw) if annual_income_raw not in (None, "") else 0.0
        )
    except (TypeError, ValueError):
        annual_income = 0.0

    if aml_status == "flagged":
        return escalate_to_compliance.invoke({
            "session_id": session_id,
            "reason": "AML flagged; escalated to compliance.",
            "severity": "high",
        })
    if aml_status == "review":
        return queue_for_review.invoke({
            "session_id": session_id,
            "reason": "AML elevated risk requires compliance review.",
            "priority": "urgent",
        })
    if fraud_status in {"flagged", "rejected"} and fraud_score >= 60:
        return reject_application.invoke({
            "session_id": session_id,
            "reason": "Fraud flagged with high risk score.",
        })
    if fraud_score >= 80:
        return reject_application.invoke({
            "session_id": session_id,
            "reason": "Fraud risk score is critical.",
        })
    if source_of_funds in {"cryptocurrency", "cash"} and annual_income > 1_000_000:
        return queue_for_review.invoke({
            "session_id": session_id,
            "reason": "High-income cash/crypto source requires manual review.",
            "priority": "urgent",
        })
    if fraud_score >= 60:
        return queue_for_review.invoke({
            "session_id": session_id,
            "reason": "Fraud risk score is elevated and needs manual review.",
            "priority": "normal",
        })
    if (
        fraud_score <= 59
        and aml_status == "clear"
        and state.get("aml_completed")
        and fraud_status not in _FRAUD_BLOCKING
    ):
        return approve_account.invoke({
            "session_id": session_id,
            "reason": "Low fraud risk and AML clear; proceed to activation.",
        })
    if aml_status in {"pending", "checking"} or not state.get("aml_completed"):
        return queue_for_review.invoke({
            "session_id": session_id,
            "reason": "AML screening still in progress.",
            "priority": "normal",
        })
    if state.get("doc_failure_type"):
        return request_additional_docs.invoke({
            "session_id": session_id,
            "reason": "Document verification requires resubmission.",
            "doc_list": ["pan", "aadhaar", "selfie"],
        })
    return None


# ---------------------------------------------------------------------------
# SECTION 4 — Agentic Node
# ---------------------------------------------------------------------------


async def _run_decision_agent(state: OnboardingState) -> dict[str, Any]:
    session_id = state.get("audit_session_id") or state.get("session_id") or "unknown"
    fraud_score = state.get("fraud_risk_score") or 0
    aml_status = state.get("aml_status")
    fraud_status = (state.get("fraud_status") or "").lower()

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
    print(
        f"[DEBUG][decision] session={session_id} fraud_score={fraud_score} "
        f"fraud_status={fraud_status} aml_status={aml_status}"
    )

    # Guard: if session already advanced to otp_verification or complete
    # (e.g. from a parallel request), skip re-running the decision.
    current_stage = state.get("stage", "")
    if current_stage in {"otp_verification", "complete"}:
        print(f"[DEBUG][decision] already_at_{current_stage}_skip session={session_id}")
        return {}

    # 1. Log entry
    _audit.log_event(
        session_id,
        "decision_agent_start",
        input_data=input_snapshot,
        metadata={
            "audit_session_id": state.get("audit_session_id") or session_id,
            "workflow_stage": state.get("stage") or "decision_agent",
            "decision_source": "pending",
        },
    )

    # 2. Deterministic rule gate to avoid low-risk regressions.
    # Keep LLM as secondary for ambiguous cases only.
    deterministic_result = resolve_deterministic_decision(state)

    decision_source = "llm"
    tool_result = None
    if deterministic_result is not None:
        tool_result = deterministic_result
        decision_source = "deterministic"

    context_string = ""
    llm_with_tools = None
    if tool_result is None:
        context_string = _build_decision_context(state)
        llm = AgentLLM().get_llm("decision_agent")
        llm_with_tools = llm.bind_tools(ALL_DECISION_TOOLS, tool_choice="required")

    # 4. Bind tools to LLM (when LLM path)
    # 5. Agentic loop (max 3 iterations)
    messages: list = [
        SystemMessage(content=DECISION_AGENT_SYSTEM_PROMPT),
        HumanMessage(content=context_string),
    ]

    if tool_result is None:
        for _ in range(3):
            response = await llm_with_tools.ainvoke(messages)
            messages.append(response)

            if not response.tool_calls:
                break

            tc = response.tool_calls[0]
            tool_name = tc.get("name", "")
            raw_args = tc.get("args") or {}
            if not isinstance(raw_args, dict):
                raw_args = {}

            if tool_name not in _TOOL_MAP:
                tool_result = queue_for_review.invoke({
                    "session_id": session_id,
                    "reason": f"Unknown tool requested by model: {tool_name}",
                    "priority": "normal",
                })
                break

            normalized_args: dict[str, Any] = dict(raw_args)
            normalized_args.setdefault("session_id", session_id)

            if tool_name in {
                "approve_account",
                "queue_for_review",
                "reject_application",
                "request_additional_docs",
                "escalate_to_compliance",
            }:
                reason = normalized_args.get("reason")
                if not isinstance(reason, str) or not reason.strip():
                    normalized_args["reason"] = "Automated decision with incomplete rationale from model output."

            if tool_name == "request_additional_docs":
                doc_list = normalized_args.get("doc_list")
                if not isinstance(doc_list, list) or not doc_list:
                    normalized_args["doc_list"] = ["Identity proof document"]

            if tool_name == "queue_for_review":
                priority = normalized_args.get("priority")
                if priority not in {"normal", "urgent"}:
                    normalized_args["priority"] = "normal"

            if tool_name == "escalate_to_compliance":
                severity = normalized_args.get("severity")
                if severity not in {"low", "medium", "high", "critical"}:
                    normalized_args["severity"] = "high"

            print(f"[DEBUG][decision] llm_tool={tool_name} raw_args={raw_args} session={session_id}")

            try:
                result = _TOOL_MAP[tool_name].invoke(normalized_args)
            except Exception as exc:
                logger.warning(
                    "Decision tool invocation failed for %s (session %s): %s",
                    tool_name,
                    session_id,
                    exc,
                )
                print(f"[DEBUG][decision] tool_invoke_error tool={tool_name} err={exc} session={session_id}")
                tool_result = queue_for_review.invoke({
                    "session_id": session_id,
                    "reason": f"Tool invocation failed for {tool_name}: {exc}",
                    "priority": "normal",
                })
                break

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
        print(f"[DEBUG][decision] no_tool_result_fallback session={session_id}")
        tool_result = queue_for_review.invoke({
            "session_id": session_id,
            "reason": "LLM did not invoke any tool — routed to manual review",
            "priority": "normal",
        })

    if tool_result.get("action") == "approve":
        fraud_status_lower = (state.get("fraud_status") or "").lower()
        if (
            aml_status != "clear"
            or not state.get("aml_completed")
            or fraud_status_lower in _FRAUD_BLOCKING
        ):
            print(
                f"[DEBUG][decision] approve_blocked_pending_checks session={session_id} "
                f"aml_status={aml_status} fraud_status={state.get('fraud_status')}"
            )
            tool_result = {
                "action": "hold_pending_checks",
                "stage": "fraud_check",
                "progress": 80,
                "decision_reason": "Final decision blocked until AML and fraud checks are clear.",
                "user_message": "We are finalizing AML and fraud checks before account activation.",
            }

    if tool_result.get("action") == "approve":
        from agents.decision.otp_service import (
            clear_otp,
            generate_otp,
            send_otp_email,
            mask_email,
            otp_recently_sent,
        )

        email_id = state.get("email_id") or ""
        full_name = state.get("full_name") or "User"
        print(f"[DEBUG][decision] approve_path session={session_id} email={email_id}")

        masked = mask_email(email_id)
        if otp_recently_sent(session_id, window_seconds=180):
            print(f"[DEBUG][decision] otp_already_sent_recently session={session_id} email={masked}")
            tool_result["stage"] = "otp_verification"
            tool_result["user_message"] = {
                "type": "OTP_REQUESTED",
                "channel": "chatbot",
                "payload": {
                    "message": (
                        "sending otp through email for activation.\n"
                        f"A code was already sent recently to {masked}. "
                        "Please enter it, or type 'resend code' if needed."
                    ),
                    "inputType": "otp",
                    "otpLength": 4,
                    "expiresInMinutes": 10,
                },
            }
        else:
            await send_activation_email(
                session_id=session_id,
                email=email_id,
                full_name=full_name,
                account_id=session_id[:8].upper(),
                account_type=state.get("account_type", "Savings"),
            )

            otp_code = generate_otp(session_id)
            if otp_code == "ACTIVE_EXISTS":
                print(f"[DEBUG][decision] otp_already_active session={session_id}")
                tool_result["stage"] = "otp_verification"
                tool_result["user_message"] = {
                    "type": "OTP_REQUESTED",
                    "channel": "chatbot",
                    "payload": {
                        "message": (
                            "sending otp through email for activation.\n"
                            f"An activation code was already sent to {masked}. "
                            "Please enter it, or type 'resend code' if needed."
                        ),
                        "inputType": "otp",
                        "otpLength": 4,
                        "expiresInMinutes": 10,
                    },
                }
            elif otp_code is None:
                print(f"[DEBUG][decision] otp_generate_failed_or_rate_limited session={session_id}")
                tool_result["stage"] = "otp_verification"
                tool_result["user_message"] = {
                    "type": "OTP_REQUESTED",
                    "channel": "chatbot",
                    "payload": {
                        "message": (
                            "sending otp through email for activation.\n"
                            "We're unable to send a new activation code right now due to resend limits. "
                            "Please wait a few minutes and type 'resend code'."
                        ),
                        "inputType": "otp",
                        "otpLength": 4,
                        "expiresInMinutes": 10,
                    },
                }
            else:
                email_sent = await send_otp_email(session_id, email_id, otp_code)

                if email_sent:
                    print(f"[DEBUG][decision] otp_email_sent session={session_id} email={masked}")
                    otp_msg = {
                        "type": "OTP_REQUESTED",
                        "channel": "chatbot",
                        "payload": {
                            "message": (
                                "sending otp through email for activation.\n"
                                "We've reached the final stage of setting up your account!\n"
                                f"We have sent a 4-digit Activation Code to your email {masked}.\n"
                                "Please enter the code to activate your account."
                            ),
                            "inputType": "otp",
                            "otpLength": 4,
                            "expiresInMinutes": 10,
                        },
                    }
                    tool_result["user_message"] = otp_msg
                    tool_result["stage"] = "otp_verification"
                else:
                    print(f"[DEBUG][decision] otp_email_send_failed session={session_id} email={masked}")
                    clear_otp(session_id)
                    tool_result["stage"] = "otp_verification"
                    tool_result["user_message"] = {
                        "type": "OTP_REQUESTED",
                        "channel": "chatbot",
                        "payload": {
                            "message": (
                                "sending otp through email for activation.\n"
                                f"We could not deliver your activation code to {masked}. "
                                "Please type 'resend code' to try again."
                            ),
                            "inputType": "otp",
                            "otpLength": 4,
                            "expiresInMinutes": 10,
                        },
                    }

    if (aml_status or "").lower() == "flagged" and tool_result.get("action") in {
        "reject",
        "escalate",
    }:
        from agents.aml.aml_user_report import build_aml_flag_user_message

        tool_result["user_message"] = await build_aml_flag_user_message(state)

    user_message = tool_result.get("user_message")
    if isinstance(user_message, dict):
        payload = user_message.get("payload") or {}
        text_message = payload.get("message") or "Please continue with the next step."
        messages = [
            {
                "role": "assistant",
                "text": text_message,
                "type": user_message.get("type", "GENERIC"),
                "payload": payload,
            }
        ]
    else:
        messages = [
            {
                "role": "assistant",
                "text": str(user_message or "Please continue with the next step."),
            }
        ]

    decision_reason = tool_result.get("decision_reason") or "Decision completed"
    decision_action = tool_result.get("action") or "queue_for_review"
    stage = tool_result.get("stage") or "manual_review"
    progress = int(tool_result.get("progress") or 85)

    update: dict[str, Any] = {
        "stage": stage,
        "decision_reason": decision_reason,
        "decision_action": decision_action,
        "progress": progress,
        "admin_override": False,
        "messages": messages,
    }
    print(
        f"[DEBUG][decision] final_update session={session_id} stage={stage} "
        f"action={decision_action} progress={progress}"
    )

    if "pending_docs" in tool_result:
        update["pending_docs"] = tool_result["pending_docs"]

    if stage in _FAQ_INVITE_STAGES:
        update["messages"] = _append_faq_invite(update.get("messages") or [])

    # 6. Store interaction in memory
    _memory.store_interaction(
        session_id=session_id,
        agent_name="decision_agent",
        input_data={
            **input_snapshot,
            "context_string": context_string,
            "decision_source": decision_source,
        },
        output_data={
            "action": decision_action,
            "reason": decision_reason,
            "stage": stage,
            "progress": progress,
        },
        risk_score=fraud_score,
        decision=decision_action,
        metadata={
            "audit_session_id": state.get("audit_session_id") or session_id,
            "workflow_stage": state.get("stage") or "decision_agent",
            "outcome_stage": stage,
            "aml_status": aml_status,
            "fraud_status": state.get("fraud_status"),
            "decision_source": decision_source,
        },
        event_type="decision_outcome",
    )

    # 7. Log exit
    _audit.log_event(
        session_id,
        "decision_agent_complete",
        output_data={
            "action": decision_action,
            "stage": stage,
            "reason": decision_reason,
            "policy_excerpt": retrieve_as_context(
                _decision_rag_query(state, fraud_score, aml_status),
                top_k=2,
            ),
        },
        decision=decision_action,
        metadata={
            "audit_session_id": state.get("audit_session_id") or session_id,
            "workflow_stage": state.get("stage") or "decision_agent",
            "outcome_stage": stage,
            "decision_source": decision_source,
            "aml_status": aml_status,
            "fraud_status": state.get("fraud_status"),
        },
    )

    return update


async def decision_agent_node(state: OnboardingState) -> dict[str, Any]:
    """Async decision node with 15-second timeout. Falls back to manual_review."""
    session_id = state.get("audit_session_id", "unknown")
    timeout_seconds = 60.0 if _is_low_risk_approval_candidate(state) else 15.0
    try:
        return await asyncio.wait_for(_run_decision_agent(state), timeout=timeout_seconds)
    except asyncio.TimeoutError:
        logger.warning("Decision agent timed out for session %s", session_id)
        from agents.decision.otp_service import otp_recently_sent, mask_email

        if _is_low_risk_approval_candidate(state) and otp_recently_sent(session_id):
            masked = mask_email(state.get("email_id") or "")
            return {
                "stage": "otp_verification",
                "decision_reason": "Approval path completed; OTP already sent.",
                "decision_action": "approve",
                "progress": 95,
                "admin_override": False,
                "messages": [
                    {
                        "role": "assistant",
                        "type": "OTP_REQUESTED",
                        "channel": "chatbot",
                        "payload": {
                            "message": (
                                "sending otp through email for activation.\n"
                                f"We have sent a 4-digit Activation Code to your email {masked}.\n"
                                "Please enter the code to activate your account."
                            ),
                            "inputType": "otp",
                            "otpLength": 4,
                            "expiresInMinutes": 10,
                        },
                    }
                ],
            }

        _audit.log_event(
            session_id,
            "decision_agent_error",
            metadata={
                "error": f"Timeout after {timeout_seconds:.0f} seconds",
                "audit_session_id": state.get("audit_session_id") or session_id,
                "workflow_stage": state.get("stage") or "decision_agent",
                "decision_source": "llm",
            },
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
            metadata={
                "error": str(exc),
                "audit_session_id": state.get("audit_session_id") or session_id,
                "workflow_stage": state.get("stage") or "decision_agent",
                "decision_source": "llm",
            },
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
