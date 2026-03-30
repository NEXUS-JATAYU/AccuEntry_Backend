import logging
import os
import asyncio
import json
import uuid
import hashlib
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

# Load .env before any app submodule reads ACCUVERIFY_URL / DB_* / MONGO_*.
load_dotenv()

from fastapi import FastAPI, Form, UploadFile, File, Depends, HTTPException
import httpx
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage, SystemMessage
from schemas.chat import (
    ChatRequest,
    ChatResponse,
    SessionDetailsResponse,
    SessionDetailsUpdateRequest,
    SessionDetailsUpdateResponse,
)
from supervisor import onboarding_graph
from state import OnboardingState
from sqlalchemy.orm import Session
from core.database import engine, Base, get_db
from core.http_client_pool import close_http_client, get_http_client
from agents.aml.aml_screening import build_aml_graph
from agents.data_capture.data_capture_validators import (
    validate_name,
    validate_date,
    validate_choice,
    validate_pan,
    validate_yes_no,
    validate_amount,
    validate_mobile_number,
    validate_email,
    validate_id_proof_number,
    validate_address,
)
from memory_manager import AgentMemoryManager
import models.customer_info
import models.compliance_logs
from scripts.init_aml_indices import init_aml_indices
from core.redis_client import redis_client

# Initialize DB tables
Base.metadata.create_all(bind=engine)

# Initialize AML MongoDB indices
init_aml_indices()

logger = logging.getLogger(__name__)

app = FastAPI()
ACCUVERIFY_URL = os.getenv("ACCUVERIFY_URL", "http://127.0.0.1:9000").rstrip("/")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

sessions: dict[str, OnboardingState] = {}
aml_tasks: dict[str, asyncio.Task] = {}
aml_graph = build_aml_graph()
agent_memory = AgentMemoryManager()
MAX_SESSION_MESSAGES = int(os.getenv("MAX_SESSION_MESSAGES", "120"))
USE_REDIS_SESSIONS = os.getenv("USE_REDIS_SESSIONS", "true").lower() in {"1", "true", "yes"}
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "86400"))
HITL_TERMINAL_STAGES = {"manual_review", "pending_docs", "escalated", "rejected", "otp_verification", "complete"}
HITL_FLAG_STAGES = {"manual_review", "pending_docs", "escalated", "rejected"}

DETAILS_EDITABLE_FIELDS: tuple[str, ...] = (
    "account_type",
    "full_name",
    "dob",
    "gender",
    "marital_status",
    "pan_number",
    "nationality",
    "occupation_type",
    "annual_income",
    "source_of_funds",
    "politically_exposed",
    "mobile_number",
    "email_id",
    "id_proof_type",
    "id_proof_number",
    "address",
    "mode_of_operation",
)

DETAILS_EDITABLE_STAGE_GUARD: set[str] = {"data_capture", "doc_verification"}

DETAILS_CHOICES: dict[str, list[str]] = {
    "gender": ["Male", "Female", "Third Gender"],
    "marital_status": ["Married", "Unmarried", "Others"],
    "occupation_type": ["Pvt. Sector", "Govt", "Business", "Student", "Retired", "Other"],
    "source_of_funds": ["Salary", "Business Income", "Agriculture", "Investment", "Pension", "Others"],
    "politically_exposed": ["Yes", "No", "Related to one"],
    "id_proof_type": ["Passport", "Voter ID", "Driving Licence", "Aadhaar", "NREGA Job Card"],
    "account_type": ["Savings", "Current", "Fixed Deposit", "Recurring Deposit"],
    "mode_of_operation": ["Self", "Either or Survivor", "Former or Survivor", "Jointly Operated"],
    "nationality": ["Yes", "No"],
}

print(f"Using AccuVerify URL: {ACCUVERIFY_URL}")


def _reward_for_outcome(outcome: str, otp_no_rework: bool) -> float:
    base = {
        "complete": 1.0,
        "otp_verification": 0.7,
        "manual_review": 0.35,
        "pending_docs": 0.25,
        "escalated": 0.2,
        "rejected": 0.1,
    }.get(outcome, 0.0)
    if otp_no_rework:
        base += 0.15
    return round(min(1.0, max(0.0, base)), 4)


def _emit_decision_feedback(
    *,
    state: OnboardingState,
    outcome: str,
    otp_no_rework: bool,
    source: str,
) -> None:
    session_id = state.get("audit_session_id") or state.get("session_id")
    if not session_id:
        return
    reward = _reward_for_outcome(outcome, otp_no_rework=otp_no_rework)
    agent_memory.store_feedback(
        agent_name="decision_agent",
        session_id=session_id,
        outcome=outcome,
        otp_no_rework=otp_no_rework,
        reward_score=reward,
        source=source,
        metadata={
            "workflow_stage": state.get("stage") or outcome,
            "decision_action": state.get("decision_action") or "",
            "aml_status": state.get("aml_status") or "",
            "fraud_status": state.get("fraud_status") or "",
        },
    )


def _session_cache_key(session_id: str) -> str:
    return f"accuentry:session:{session_id}"


async def _save_session_cache(session_id: str, state: OnboardingState) -> None:
    if not USE_REDIS_SESSIONS or redis_client is None:
        return
    try:
        await redis_client.set(_session_cache_key(session_id), json.dumps(state), ex=SESSION_TTL_SECONDS)
    except Exception as exc:
        print(f"[DEBUG][redis] save_failed sid={session_id} err={exc}")


async def _load_session_cache(session_id: str) -> OnboardingState | None:
    if not USE_REDIS_SESSIONS or redis_client is None:
        return None
    try:
        raw = await redis_client.get(_session_cache_key(session_id))
        if not raw:
            return None
        cached = json.loads(raw)
        if not isinstance(cached, dict):
            return None
        # Rehydrate with defaults so missing keys from older sessions do not break flow.
        return {**_initial_onboarding_state(session_id), **cached}
    except Exception as exc:
        print(f"[DEBUG][redis] load_failed sid={session_id} err={exc}")
        return None


def _initial_onboarding_state(session_id: str) -> OnboardingState:
    import uuid as _uuid
    return {
        "session_id": session_id,
        "messages": [],
        "stage": "data_capture",
        "full_name": None,
        "dob": None,
        "gender": None,
        "marital_status": None,
        "pan_number": None,
        "nationality": None,
        "occupation_type": None,
        "annual_income": None,
        "source_of_funds": None,
        "politically_exposed": None,
        "mobile_number": None,
        "email_id": None,
        "id_proof_type": None,
        "id_proof_number": None,
        "address": None,
        "account_type": None,
        "mode_of_operation": None,
        "debit_card_required": None,
        "internet_banking": None,
        "mobile_banking": None,
        "sms_alerts": None,
        "cheque_book": None,
        "nominee_name": None,
        "nominee_relationship": None,
        "nominee_dob": None,
        "pan_verified": None,
        "aadhaar_verified": None,
        "face_verified": None,
        "kyc_status": None,
        "aml_status": "pending",
        "aml_raw_results": None,
        "aml_risk_score": None,
        "aml_in_background": False,
        "aml_completed": False,
        "fraud_status": None,
        "fraud_risk_score": None,
        "fraud_signals": [],
        "fraud_reasoning": None,
        "metadata": {},
        "progress": 0,
        "requires_upload": False,
        "capture_target": None,
        "capture_candidate": None,
        "capture_error": None,
        "doc_failure_type": None,
        # Decision agent fields
        "decision_reason": None,
        "decision_action": None,
        "pending_docs": [],
        "admin_override": False,
        "audit_session_id": str(_uuid.uuid4()),
        # Upstream signal fields for decision agent
        "video_kyc_status": None,  # TODO: not yet implemented — video_kyc
        "risk_model_label": None,  # TODO: not yet implemented — risk_analysis
        "risk_model_confidence": None,  # TODO: not yet implemented — risk_analysis
        "kyc_data": None,
    }


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _details_payload_from_state(state: OnboardingState) -> dict[str, str | None]:
    payload: dict[str, str | None] = {}
    for key in DETAILS_EDITABLE_FIELDS:
        value = state.get(key)
        payload[key] = None if value is None else str(value)
    return payload


async def _ensure_session_state(session_id: str) -> OnboardingState:
    if session_id in sessions:
        return sessions[session_id]

    cached = await _load_session_cache(session_id)
    if cached is not None:
        sessions[session_id] = cached
        return sessions[session_id]

    raise HTTPException(status_code=404, detail="Session not found")


def _validate_details_field(field: str, value: object) -> tuple[bool, str]:
    candidate = "" if value is None else str(value)

    if field == "full_name":
        return validate_name(candidate)
    if field == "dob":
        return validate_date(candidate)
    if field in DETAILS_CHOICES:
        return validate_choice(candidate, DETAILS_CHOICES[field], field.replace("_", " "))
    if field == "pan_number":
        return validate_pan(candidate)
    if field == "annual_income":
        return validate_amount(candidate, "annual income")
    if field == "mobile_number":
        return validate_mobile_number(candidate)
    if field == "email_id":
        return validate_email(candidate)
    if field == "id_proof_number":
        return validate_id_proof_number(candidate)
    if field == "address":
        return validate_address(candidate)

    return False, f"Unsupported field: {field}"


def _owner_initials(full_name: str | None) -> str:
    parts = [p for p in (full_name or "").strip().split() if p]
    if not parts:
        return "NA"
    if len(parts) == 1:
        return parts[0][:2].upper()
    return (parts[0][0] + parts[-1][0]).upper()


def _hitl_status_from_stage(stage: str) -> str:
    if stage == "rejected":
        return "Inactive"
    if stage in {"otp_verification", "complete"}:
        return "Active"
    return "Pending"


def _hitl_compliance(stage: str) -> str:
    if stage in HITL_FLAG_STAGES:
        return "Non-Compliant"
    if stage in {"otp_verification", "complete"}:
        return "Compliant"
    return "Pending"


def _build_hitl_report(state: OnboardingState) -> tuple[bool, str]:
    stage = state.get("stage") or "data_capture"
    decision_action = state.get("decision_action") or "undecided"
    decision_reason = state.get("decision_reason") or "No decision reason provided"
    fraud_score = state.get("fraud_risk_score")
    aml_status = state.get("aml_status") or "pending"
    signals = state.get("fraud_signals") or []
    signals_text = ", ".join(signals[:4]) if signals else "none"

    if stage in HITL_FLAG_STAGES:
        report = (
            f"Flag raised for manual attention. Stage={stage}. Decision={decision_action}. "
            f"Reason={decision_reason}. Fraud score={fraud_score}. AML={aml_status}. "
            f"Signals={signals_text}."
        )
        return True, report

    report = (
        f"Valid onboarding flow. Stage={stage}. Decision={decision_action}. "
        f"Reason={decision_reason}. Fraud score={fraud_score}. AML={aml_status}. "
        "Account is progressing normally for activation."
    )
    return False, report


def _build_hitl_case(session_id: str, state: OnboardingState) -> dict:
    stage = state.get("stage") or "data_capture"
    flagged, report = _build_hitl_report(state)
    audit_id = state.get("audit_session_id") or session_id
    updated_at = ((state.get("metadata") or {}).get("updated_at") or _iso_now())
    alerts = len(state.get("fraud_signals") or []) + (1 if flagged else 0)
    decision_action = state.get("decision_action") or "in_progress"

    return {
        "id": str(audit_id)[:12].upper(),
        "session_id": session_id,
        "audit_session_id": audit_id,
        "obligation": report,
        "status": _hitl_status_from_stage(stage),
        "module": "Decision Engine" if stage in HITL_TERMINAL_STAGES else "Onboarding Pipeline",
        "jurisdiction": f"AML {state.get('aml_status') or 'pending'}",
        "alerts": alerts,
        "compliance": _hitl_compliance(stage),
        "owner": _owner_initials(state.get("full_name")),
        "ownerColor": "#2563eb" if flagged else "#16a34a",
        "due": updated_at.replace("T", " ")[:19],
        "flagged": flagged,
        "report": report,
        "stage": stage,
        "decision_action": decision_action,
        "decision_reason": state.get("decision_reason"),
        "fraud_risk_score": state.get("fraud_risk_score"),
        "fraud_signals": state.get("fraud_signals") or [],
        "aml_status": state.get("aml_status"),
        "progress": state.get("progress", 0),
        "updated_at": updated_at,
        "full_name": state.get("full_name"),
        "email_id": state.get("email_id"),
    }


async def _collect_all_session_states() -> dict[str, OnboardingState]:
    merged: dict[str, OnboardingState] = dict(sessions)
    if not USE_REDIS_SESSIONS or redis_client is None:
        return merged

    try:
        async for key in redis_client.scan_iter(match="accuentry:session:*"):
            sid = key.split(":")[-1]
            if sid in merged:
                continue
            raw = await redis_client.get(key)
            if not raw:
                continue
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                merged[sid] = {**_initial_onboarding_state(sid), **parsed}
    except Exception as exc:
        print(f"[DEBUG][hitl] redis_collect_failed err={exc}")

    return merged


def _should_start_aml(state: OnboardingState) -> bool:
    stage = state.get("stage") or ""
    kyc_approved = state.get("kyc_status") == "approved"
    metadata = state.get("metadata") or {}
    # Some flows can reach fraud_check with kyc_status not persisted; allow AML start there.
    if not kyc_approved and stage not in {"fraud_check", "aml_screening"}:
        return False
    if metadata.get("aml_run_started"):
        return False
    if state.get("aml_completed"):
        return False
    if state.get("aml_in_background"):
        return False
    return state.get("aml_status") in (None, "pending")


async def _run_aml_background(session_id: str) -> None:
    try:
        current = sessions.get(session_id)
        if not current:
            return

        aml_input: OnboardingState = {
            **current,
            "messages": list(current["messages"]),
            "stage": "aml_screening",
        }
        result = await aml_graph.ainvoke(aml_input, config={"recursion_limit": 50})

        latest = sessions.get(session_id)
        if not latest:
            return

        pre_decision_stages = {"kyc_approval", "aml_screening", "fraud_check"}
        latest_stage = latest.get("stage")
        can_apply_aml_stage = latest_stage in pre_decision_stages

        new_messages = result.get("messages", [])
        old_messages = aml_input.get("messages", [])
        aml_appended_messages = new_messages[len(old_messages):] if len(new_messages) >= len(old_messages) else []
        if not can_apply_aml_stage:
            aml_appended_messages = []

        resolved_aml_status = result.get("aml_status", latest.get("aml_status", "pending"))

        updated: OnboardingState = {
            **latest,
            "aml_raw_results": result.get("aml_raw_results", latest.get("aml_raw_results")),
            "aml_risk_score": result.get("aml_risk_score", latest.get("aml_risk_score")),
            "aml_status": resolved_aml_status,
            "stage": result.get("stage", latest_stage) if can_apply_aml_stage else latest_stage,
            "messages": latest.get("messages", []) + aml_appended_messages,
            "aml_in_background": False,
            "aml_completed": resolved_aml_status in ("clear", "flagged"),
            "metadata": {**(latest.get("metadata") or {}), "updated_at": _iso_now()},
        }

        sessions[session_id] = _trim_messages(updated)
    except Exception as exc:
        latest = sessions.get(session_id)
        if latest:
            sessions[session_id] = _trim_messages({
                **latest,
                "aml_in_background": False,
                "aml_status": "flagged",
                "aml_completed": True,
                "stage": "manual_review",
                "messages": (latest.get("messages") or []) + [
                    {
                        "role": "assistant",
                        "text": "We encountered an issue while running AML screening. Your application has been routed for manual review.",
                    }
                ],
                "metadata": {**(latest.get("metadata") or {}), "updated_at": _iso_now()},
            })
        print(f"Background AML task failed for {session_id}: {exc}")
    finally:
        aml_tasks.pop(session_id, None)


def _start_aml_if_needed(session_id: str) -> None:
    state = sessions.get(session_id)
    if not state or not _should_start_aml(state):
        return

    existing = aml_tasks.get(session_id)
    if existing and not existing.done():
        return

    sessions[session_id] = {
        **state,
        "aml_status": "checking",
        "aml_in_background": True,
        "metadata": {
            **(state.get("metadata") or {}),
            "updated_at": _iso_now(),
            "aml_run_started": True,
            "aml_started_at": _iso_now(),
        },
    }
    aml_tasks[session_id] = asyncio.create_task(_run_aml_background(session_id))


def _last_assistant_text(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "assistant":
            if m.get("type") and isinstance(m.get("payload"), dict):
                return json.dumps({
                    "type": m.get("type"),
                    "channel": m.get("channel", "chatbot"),
                    "payload": m.get("payload"),
                })
            return (m.get("text") or "").strip()
    return ""


def _trim_messages(state: OnboardingState) -> OnboardingState:
    msgs = state.get("messages", [])
    if len(msgs) <= MAX_SESSION_MESSAGES:
        return state
    return {**state, "messages": list(msgs[-MAX_SESSION_MESSAGES:])}


def _fallback_assistant_message(stage: str, requires_upload: bool) -> str:
    if requires_upload:
        return (
            "Please use the upload area below to add your PAN, Aadhaar card, and selfie "
            "when you are ready."
        )
    if stage == "data_capture":
        return "Got it. Reply in the chat when you are ready to continue."
    return "Continue when you are ready, or send another message to check status."


def _ui_step_and_label(stage: str) -> tuple[int, str]:
    mapping: dict[str, tuple[int, str]] = {
        "data_capture": (1, "Detail Capture"),
        "doc_verification": (2, "Identity Verification"),
        "kyc_approval": (3, "KYC review"),
        "aml_screening": (3, "AML Screening"),
        "fraud_check": (4, "Fraud Check"),
        "manual_review": (5, "Manual Review"),
        "pending_docs": (5, "Pending Documents"),
        "escalated": (5, "Compliance Escalation"),
        "otp_verification": (5, "Account Activation"),
        "complete": (5, "Account Activated"),
        "rejected": (5, "Application rejected"),
    }
    return mapping.get(stage, (1, "Detail Capture"))

def _save_state_to_db(state: OnboardingState, db: Session) -> None:
    from models.customer_info import CustomerDetails
    from datetime import datetime

    metadata = state.get("metadata") or {}
    if not metadata.get("details_confirmation_saved"):
        return
    
    required_fields = ["full_name", "mobile_number", "email_id", "address", "occupation_type", "pan_number", "dob"]
    for f in required_fields:
        if not state.get(f):
            return
            
    try:
        c_dob_date = datetime.strptime(state["dob"], "%Y-%m-%d").date()
    except Exception:
        return
        
    try:
        cust = db.query(CustomerDetails).filter(CustomerDetails.c_phone_number == state["mobile_number"]).first()
        if not cust:
            cust = CustomerDetails(c_phone_number=state["mobile_number"])
            db.add(cust)
            
        cust.c_name = state["full_name"]
        cust.c_account_type = state.get("account_type") or "Savings"
        cust.c_email = state["email_id"]
        cust.c_address = state["address"]
        cust.c_occupation = state["occupation_type"]
        cust.c_pan = state["pan_number"]
        cust.c_dob = c_dob_date
        
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"Error saving to DB: {e}")


def _upsert_account_type_capture(state: OnboardingState, db: Session) -> None:
    from models.compliance_logs import AccountTypeCapture

    session_id = state.get("session_id")
    if not session_id:
        return

    try:
        row = db.query(AccountTypeCapture).filter(AccountTypeCapture.session_id == session_id).first()
        if not row:
            row = AccountTypeCapture(session_id=session_id)
            db.add(row)

        row.account_type = str(state.get("account_type") or row.account_type or "Unknown")

        db.commit()
    except Exception as exc:
        db.rollback()
        print(f"Error upserting account_type_capture for {session_id}: {exc}")


def _upsert_compliance_stage_alert(state: OnboardingState, db: Session) -> None:
    from models.compliance_logs import ComplianceStageAlert

    session_id = state.get("session_id")
    if not session_id:
        return

    stage = state.get("stage") or "data_capture"
    flagged, report = _build_hitl_report(state)
    signals = state.get("fraud_signals") or []
    alert_count = len(signals) + (1 if flagged else 0)

    alert_details = {
        "fraud_signals": signals,
        "decision_action": state.get("decision_action"),
        "decision_reason": state.get("decision_reason"),
        "fraud_risk_score": state.get("fraud_risk_score"),
        "aml_status": state.get("aml_status"),
    }

    try:
        row = (
            db.query(ComplianceStageAlert)
            .filter(
                ComplianceStageAlert.session_id == session_id,
                ComplianceStageAlert.stage == stage,
            )
            .first()
        )
        if not row:
            row = ComplianceStageAlert(session_id=session_id, stage=stage)
            db.add(row)

        row.audit_session_id = str(state.get("audit_session_id")) if state.get("audit_session_id") else None
        row.alert_count = alert_count
        row.alert_summary = report
        row.alert_details_json = alert_details
        row.overall_flagged = flagged

        db.commit()
    except Exception as exc:
        db.rollback()
        print(f"Error upserting compliance stage alert for {session_id}: {exc}")


def _persist_session_tracking(state: OnboardingState, db: Session) -> None:
    _upsert_account_type_capture(state, db)
    _upsert_compliance_stage_alert(state, db)
    _upsert_llm_decision_summary_log(state, db)


def _generate_llm_decision_summary(state: OnboardingState) -> tuple[str, bool]:
    stage = state.get("stage") or "data_capture"
    decision_action = state.get("decision_action") or "undecided"
    decision_reason = state.get("decision_reason") or "No reason was provided"
    fraud_score = state.get("fraud_risk_score")
    aml_status = state.get("aml_status") or "pending"
    fraud_signals = state.get("fraud_signals") or []

    fallback = (
        f"- Stage: {stage}\n"
        f"- Decision: {decision_action}\n"
        f"- Reason: {decision_reason}\n"
        f"- Risk snapshot: fraud_score={fraud_score}, aml_status={aml_status}, signals={', '.join(fraud_signals[:4]) or 'none'}"
    )

    try:
        from llm_config import AgentLLM

        payload = {
            "stage": stage,
            "decision_action": decision_action,
            "decision_reason": decision_reason,
            "fraud_score": fraud_score,
            "aml_status": aml_status,
            "fraud_signals": fraud_signals,
        }
        llm = AgentLLM().get_llm("decision_agent")
        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are writing customer-friendly compliance audit notes. "
                        "Return exactly 3-5 concise bullet points, each prefixed with '-'. "
                        "Do not invent details. Keep it factual and readable for operations teams."
                    )
                ),
                HumanMessage(
                    content=(
                        "Convert this decision event into human-readable bullet points:\n"
                        f"{json.dumps(payload, default=str)}"
                    )
                ),
            ]
        )
        text = str(getattr(response, "content", "") or "").strip()
        if not text:
            return fallback, False
        return text, True
    except Exception as exc:
        print(f"[DEBUG][decision_log] llm_generation_failed err={exc}")
        return fallback, False


def _upsert_llm_decision_summary_log(state: OnboardingState, db: Session) -> None:
    from models.compliance_logs import LLMDecisionLog

    log_session_id = state.get("audit_session_id") or state.get("session_id")
    session_id = state.get("session_id")
    if not log_session_id or not session_id:
        return

    stage = state.get("stage") or "data_capture"
    decision_action = state.get("decision_action") or "undecided"
    decision_reason = state.get("decision_reason") or "No reason was provided"

    try:
        existing = (
            db.query(LLMDecisionLog)
            .filter(
                LLMDecisionLog.session_id == str(log_session_id),
                LLMDecisionLog.event_type == "decision_summary",
                LLMDecisionLog.stage == stage,
                LLMDecisionLog.decision == decision_action,
            )
            .first()
        )
        if existing:
            return

        friendly_text, used_llm = _generate_llm_decision_summary(state)
        hash_payload = f"{log_session_id}|{stage}|decision_summary|{friendly_text.strip().lower()}"
        log_hash = hashlib.sha256(hash_payload.encode("utf-8")).hexdigest()

        db.add(
            LLMDecisionLog(
                session_id=str(log_session_id),
                audit_session_id=str(log_session_id),
                stage=stage,
                event_type="decision_summary",
                decision_source="llm" if used_llm else "fallback",
                decision=decision_action,
                friendly_text=friendly_text,
                log_hash=log_hash,
                input_payload_json={
                    "fraud_score": state.get("fraud_risk_score"),
                    "aml_status": state.get("aml_status"),
                    "fraud_signals": state.get("fraud_signals") or [],
                },
                output_payload_json={
                    "action": decision_action,
                    "reason": decision_reason,
                    "stage": stage,
                },
                metadata_json={
                    "source_session_id": session_id,
                    "audit_session_id": state.get("audit_session_id"),
                    "decision_reason": decision_reason,
                },
            )
        )
        db.commit()
    except Exception as exc:
        db.rollback()
        print(f"Error upserting LLM decision summary log for {log_session_id}: {exc}")


@app.get("/session/{session_id}/details", response_model=SessionDetailsResponse)
async def get_session_details(session_id: str):
    state = await _ensure_session_state(session_id)
    return SessionDetailsResponse(
        session_id=session_id,
        stage=state.get("stage", "data_capture"),
        details=_details_payload_from_state(state),
    )


@app.put("/session/{session_id}/details", response_model=SessionDetailsUpdateResponse)
async def update_session_details(
    session_id: str,
    request: SessionDetailsUpdateRequest,
    db: Session = Depends(get_db),
):
    if request.session_id != session_id:
        raise HTTPException(status_code=400, detail="Session id mismatch")

    state = await _ensure_session_state(session_id)
    stage = state.get("stage", "data_capture")

    if stage not in DETAILS_EDITABLE_STAGE_GUARD:
        raise HTTPException(status_code=409, detail=f"Details cannot be edited in stage '{stage}'")

    requested_details = request.details or {}
    if not requested_details:
        updated_noop: OnboardingState = {
            **state,
            "metadata": {
                **(state.get("metadata") or {}),
                "details_confirmation_saved": True,
                "updated_at": _iso_now(),
            },
        }
        sessions[session_id] = _trim_messages(updated_noop)
        await _save_session_cache(session_id, sessions[session_id])
        _save_state_to_db(sessions[session_id], db)
        _persist_session_tracking(sessions[session_id], db)
        return SessionDetailsUpdateResponse(
            session_id=session_id,
            stage=stage,
            details=_details_payload_from_state(sessions[session_id]),
            message="Details confirmed and saved.",
            errors={},
        )

    invalid_fields = [f for f in requested_details.keys() if f not in DETAILS_EDITABLE_FIELDS]
    if invalid_fields:
        raise HTTPException(status_code=400, detail=f"Unsupported fields: {', '.join(invalid_fields)}")

    normalized: dict[str, str] = {}
    errors: dict[str, str] = {}

    for field, value in requested_details.items():
        ok, result = _validate_details_field(field, value)
        if ok:
            normalized[field] = result
        else:
            errors[field] = result

    if errors:
        return SessionDetailsUpdateResponse(
            session_id=session_id,
            stage=stage,
            details=_details_payload_from_state(state),
            message="Validation failed. Please correct highlighted fields.",
            errors=errors,
        )

    updated: OnboardingState = {
        **state,
        **normalized,
        "metadata": {
            **(state.get("metadata") or {}),
            "details_confirmation_saved": True,
            "updated_at": _iso_now(),
        },
    }
    sessions[session_id] = _trim_messages(updated)
    await _save_session_cache(session_id, sessions[session_id])
    _save_state_to_db(sessions[session_id], db)
    _persist_session_tracking(sessions[session_id], db)

    return SessionDetailsUpdateResponse(
        session_id=session_id,
        stage=sessions[session_id].get("stage", "data_capture"),
        details=_details_payload_from_state(sessions[session_id]),
        message="Details updated successfully.",
        errors={},
    )


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, db: Session = Depends(get_db)):
    try:
        sid = (request.session_id or "").strip() or str(uuid.uuid4())
        print(f"[DEBUG][chat] incoming sid={sid} user_input={request.user_input!r}")
        if sid not in sessions:
            cached = await _load_session_cache(sid)
            if cached is not None:
                print(f"[DEBUG][chat] restored_session_from_redis sid={sid}")
                sessions[sid] = cached
            else:
                print(f"[DEBUG][chat] new_session sid={sid}")
                sessions[sid] = _initial_onboarding_state(sid)
                await _save_session_cache(sid, sessions[sid])

        state: OnboardingState = {**sessions[sid], "messages": list(sessions[sid]["messages"])}
        if not state.get("audit_session_id"):
            import uuid as _uuid
            state["audit_session_id"] = str(_uuid.uuid4())
            print(f"[DEBUG][chat] audit_session_id_missing_generated sid={sid} audit={state['audit_session_id']}")
        if not state.get("session_id"):
            state["session_id"] = sid
            print(f"[DEBUG][chat] session_id_missing_set sid={sid}")
        _persist_session_tracking(state, db)
        text = (request.user_input or "").strip()
        print(
            f"[DEBUG][chat] sid={sid} stage={state.get('stage')} "
            f"audit={state.get('audit_session_id')} text_present={bool(text)}"
        )

        # Empty-input status polls should not re-run terminal/OTP flows.
        if not text and state.get("stage") in {
            "otp_verification",
            "complete",
            "manual_review",
            "pending_docs",
            "escalated",
            "rejected",
        }:
            stage = state["stage"]
            step, main_step = _ui_step_and_label(stage)
            effective_progress = 100 if stage in {"complete", "rejected"} else state.get("progress", 0)
            reply = _last_assistant_text(state.get("messages", [])) or _fallback_assistant_message(
                stage,
                state.get("requires_upload", False),
            )
            print(f"[DEBUG][chat] empty_poll_short_circuit sid={sid} stage={stage}")
            return ChatResponse(
                message=reply,
                progress=effective_progress,
                requires_upload=state.get("requires_upload", False),
                stage=stage,
                completed=stage == "complete",
                step=step,
                current_main_step=main_step,
                aml_status=state.get("aml_status", "pending"),
                aml_in_background=state.get("aml_in_background", False),
                fraud_status=state.get("fraud_status"),
                fraud_risk_score=state.get("fraud_risk_score"),
                fraud_signals=state.get("fraud_signals", []),
                fraud_reasoning=state.get("fraud_reasoning"),
                otp_required=stage == "otp_verification",
            )

        # ── OTP verification intercept (no LLM needed) ──────────
        if state["stage"] == "otp_verification" and text:
            import re as _re
            import json as _json
            import uuid as _uuid
            from datetime import datetime, timezone
            from agents.decision.otp_service import (
                verify_otp,
                send_confirmation_email,
                send_otp_email,
                generate_otp,
                mask_email,
                is_otp_locked,
                get_otp_send_count,
            )

            if text:
                state["messages"].append({"role": "user", "text": text})

            digits = _re.sub(r"\D", "", text)
            otp_session_id = state.get("audit_session_id") or sid
            print(
                f"[DEBUG][otp] sid={sid} otp_session={otp_session_id} "
                f"text={text!r} digits={digits}"
            )

            if is_otp_locked(otp_session_id):
                print(f"[DEBUG][otp] locked sid={sid} otp_session={otp_session_id}")
                reply = "Too many incorrect attempts. Please restart the activation process or contact support."
                state["messages"].append({"role": "assistant", "text": reply})
                sessions[sid] = _trim_messages(state)
                await _save_session_cache(sid, sessions[sid])
                stage = state["stage"]
                step, main_step = _ui_step_and_label(stage)
                return ChatResponse(
                    message=reply, progress=state["progress"],
                    stage=stage, step=step, current_main_step=main_step,
                    aml_status=state.get("aml_status", "pending"),
                    aml_in_background=state.get("aml_in_background", False),
                    fraud_status=state.get("fraud_status"),
                    fraud_risk_score=state.get("fraud_risk_score"),
                    fraud_signals=state.get("fraud_signals", []),
                    fraud_reasoning=state.get("fraud_reasoning"),
                    otp_required=True,
                )

            lowered = text.lower()
            resend_requested = any(
                token in lowered
                for token in (
                    "resend",
                    "new code",
                    "send again",
                    "didn't receive",
                    "did not receive",
                )
            )
            if resend_requested:
                print(f"[DEBUG][otp] resend_requested sid={sid} otp_session={otp_session_id}")
                email_id = state.get("email_id") or ""
                masked = mask_email(email_id)
                otp_code = generate_otp(otp_session_id)
                if otp_code is None:
                    print(f"[DEBUG][otp] resend_rate_limited sid={sid} otp_session={otp_session_id}")
                    resend_message = (
                        "You have reached the OTP resend limit. "
                        "Please wait a few minutes and try again."
                    )
                elif not email_id:
                    print(f"[DEBUG][otp] resend_missing_email sid={sid}")
                    resend_message = "We do not have a valid email on file for this application. Please contact support."
                else:
                    sent = await send_otp_email(otp_session_id, email_id, otp_code)
                    if sent:
                        print(f"[DEBUG][otp] resend_sent sid={sid} email={masked}")
                        resend_message = (
                            f"A new 4-digit activation code has been sent to {masked}. "
                            "Please enter it here to activate your account."
                        )
                    else:
                        print(f"[DEBUG][otp] resend_send_failed sid={sid} email={masked}")
                        resend_message = (
                            f"We could not send a new activation code to {masked}. "
                            "Please try again shortly or contact support."
                        )

                reply_payload = {
                    "type": "OTP_REQUESTED",
                    "channel": "chatbot",
                    "payload": {
                        "message": resend_message,
                        "inputType": "otp",
                        "otpLength": 4,
                        "expiresInMinutes": 10,
                    },
                }
                reply = _json.dumps(reply_payload)

                state["messages"].append({"role": "assistant", "text": reply})
                sessions[sid] = _trim_messages(state)
                await _save_session_cache(sid, sessions[sid])
                stage = state["stage"]
                step, main_step = _ui_step_and_label(stage)
                return ChatResponse(
                    message=reply, progress=state["progress"],
                    stage=stage, step=step, current_main_step=main_step,
                    aml_status=state.get("aml_status", "pending"),
                    aml_in_background=state.get("aml_in_background", False),
                    fraud_status=state.get("fraud_status"),
                    fraud_risk_score=state.get("fraud_risk_score"),
                    fraud_signals=state.get("fraud_signals", []),
                    fraud_reasoning=state.get("fraud_reasoning"),
                    otp_required=True,
                )

            if len(digits) != 4:
                print(f"[DEBUG][otp] invalid_digits sid={sid} digits={digits}")
                reply = "Please enter a valid 4-digit activation code."
                state["messages"].append({"role": "assistant", "text": reply})
                sessions[sid] = _trim_messages(state)
                await _save_session_cache(sid, sessions[sid])
                stage = state["stage"]
                step, main_step = _ui_step_and_label(stage)
                return ChatResponse(
                    message=reply, progress=state["progress"],
                    stage=stage, step=step, current_main_step=main_step,
                    aml_status=state.get("aml_status", "pending"),
                    aml_in_background=state.get("aml_in_background", False),
                    fraud_status=state.get("fraud_status"),
                    fraud_risk_score=state.get("fraud_risk_score"),
                    fraud_signals=state.get("fraud_signals", []),
                    fraud_reasoning=state.get("fraud_reasoning"),
                    otp_required=True,
                )

            success, msg = verify_otp(otp_session_id, digits)
            print(f"[DEBUG][otp] verify_result sid={sid} otp_session={otp_session_id} success={success} msg={msg}")

            if not success:
                state["messages"].append({"role": "assistant", "text": msg})
                sessions[sid] = _trim_messages(state)
                await _save_session_cache(sid, sessions[sid])
                stage = state["stage"]
                step, main_step = _ui_step_and_label(stage)
                return ChatResponse(
                    message=msg, progress=state["progress"],
                    stage=stage, step=step, current_main_step=main_step,
                    aml_status=state.get("aml_status", "pending"),
                    aml_in_background=state.get("aml_in_background", False),
                    fraud_status=state.get("fraud_status"),
                    fraud_risk_score=state.get("fraud_risk_score"),
                    fraud_signals=state.get("fraud_signals", []),
                    fraud_reasoning=state.get("fraud_reasoning"),
                    otp_required=not is_otp_locked(otp_session_id),
                )

            # ── OTP Success: Activate account ──────────────────
            account_id = f"ACC-{_uuid.uuid4().hex[:8].upper()}"
            full_name = state.get("full_name") or "User"
            email_id = state.get("email_id") or ""
            account_type = state.get("account_type") or "Savings"
            now = datetime.now(timezone.utc)
            now_iso = now.isoformat() + "Z"
            now_display = now.strftime("%Y-%m-%d %H:%M:%S UTC")

            await send_confirmation_email(
                otp_session_id, email_id, full_name, account_id, account_type, now_display,
            )
            print(f"[DEBUG][otp] activation_complete sid={sid} account_id={account_id}")

            activation_msg = _json.dumps({
                "type": "ACCOUNT_ACTIVATED",
                "channel": "chatbot",
                "payload": {
                    "message": "Congratulations! 🎉\nYour Account has been Activated!\nThank You For Banking With Us!",
                    "status": "ACTIVE",
                    "activatedAt": now_iso,
                    "account": {
                        "accountId": account_id,
                        "accountHolderName": full_name,
                        "accountType": account_type,
                    },
                },
            })
            print(f"[DEBUG][otp] activation_msg_content={activation_msg}")

            state["stage"] = "complete"
            state["progress"] = 100
            state["messages"].append({"role": "assistant", "text": activation_msg})
            sessions[sid] = _trim_messages(state)
            sessions[sid]["metadata"] = {**(sessions[sid].get("metadata") or {}), "updated_at": _iso_now()}
            otp_no_rework = get_otp_send_count(otp_session_id) <= 1
            _emit_decision_feedback(
                state=sessions[sid],
                outcome="complete",
                otp_no_rework=otp_no_rework,
                source="otp_verification",
            )
            await _save_session_cache(sid, sessions[sid])
            _save_state_to_db(sessions[sid], db)
            _persist_session_tracking(sessions[sid], db)

            step, main_step = _ui_step_and_label("complete")
            return ChatResponse(
                message=activation_msg, progress=100,
                stage="complete", completed=True,
                step=step, current_main_step=main_step,
                aml_status=state.get("aml_status", "pending"),
                aml_in_background=state.get("aml_in_background", False),
                fraud_status=state.get("fraud_status"),
                fraud_risk_score=state.get("fraud_risk_score"),
                fraud_signals=state.get("fraud_signals", []),
                fraud_reasoning=state.get("fraud_reasoning"),
                otp_required=False,
            )

        # ── Normal graph flow ───────────────────────────────────
        if text:
            state["messages"].append({"role": "user", "text": text})

        new_state = await onboarding_graph.ainvoke(
            state,
            config={"recursion_limit": 50},
        )

        # ── Reconcile AML background results ─────────────────────
        # The graph ran with a snapshot of the session. While it was
        # executing, the AML background task may have completed and
        # written fresh results directly to sessions[sid].  If the
        # graph's output still carries stale AML values (e.g.
        # aml_status="checking") but the live session already has the
        # real outcome, prefer the live session's AML fields.
        live = sessions.get(sid) or {}
        live_aml_done = live.get("aml_completed", False)
        graph_aml_stale = new_state.get("aml_status") in (None, "pending", "checking")
        if live_aml_done and graph_aml_stale:
            for key in (
                "aml_status", "aml_completed", "aml_in_background",
                "aml_risk_score", "aml_raw_results",
            ):
                if key in live:
                    new_state[key] = live[key]
            # Also pull back the metadata flag so AML doesn't re-trigger
            live_meta = live.get("metadata") or {}
            new_meta = new_state.get("metadata") or {}
            new_state["metadata"] = {**new_meta, **{
                k: live_meta[k] for k in ("aml_run_started", "aml_started_at")
                if k in live_meta
            }}
        # ──────────────────────────────────────────────────────────

        print(
            f"[DEBUG][chat] graph_done sid={sid} stage_before={state.get('stage')} "
            f"stage_after={new_state.get('stage')} decision_action={new_state.get('decision_action')} "
            f"fraud_score={new_state.get('fraud_risk_score')} aml_status={new_state.get('aml_status')}"
        )
        sessions[sid] = _trim_messages(new_state)
        if sessions[sid].get("stage") in {"complete", "rejected"}:
            sessions[sid]["progress"] = 100
        sessions[sid]["metadata"] = {**(sessions[sid].get("metadata") or {}), "updated_at": _iso_now()}
        await _save_session_cache(sid, sessions[sid])
        _start_aml_if_needed(sid)
        _save_state_to_db(sessions[sid], db)
        _persist_session_tracking(sessions[sid], db)

        latest_state = sessions[sid]

        terminal_outcomes = {"manual_review", "pending_docs", "escalated", "rejected"}
        if latest_state.get("stage") in terminal_outcomes:
            _emit_decision_feedback(
                state=latest_state,
                outcome=str(latest_state.get("stage")),
                otp_no_rework=False,
                source="workflow_terminal",
            )

        stage = latest_state["stage"]
        step, main_step = _ui_step_and_label(stage)
        effective_progress = 100 if stage in {"complete", "rejected"} else latest_state["progress"]
        raw_reply = _last_assistant_text(latest_state["messages"])
        reply = raw_reply or _fallback_assistant_message(
            stage,
            latest_state["requires_upload"],
        )
        return ChatResponse(
            message=reply,
            progress=effective_progress,
            requires_upload=latest_state["requires_upload"],
            stage=stage,
            completed=stage == "complete",
            step=step,
            current_main_step=main_step,
            aml_status=latest_state.get("aml_status", "pending"),
            aml_in_background=latest_state.get("aml_in_background", False),
            fraud_status=latest_state.get("fraud_status"),
            fraud_risk_score=latest_state.get("fraud_risk_score"),
            fraud_signals=latest_state.get("fraud_signals", []),
            fraud_reasoning=latest_state.get("fraud_reasoning"),
            otp_required=stage == "otp_verification",
        )
    except Exception as e:
        print(f"Error handling chat: {e}")
        raise e


@app.get("/hitl/cases")
async def get_hitl_cases(include_in_progress: bool = True, flagged_only: bool = False):
    all_states = await _collect_all_session_states()
    cases: list[dict] = []
    for sid, state in all_states.items():
        stage = state.get("stage") or "data_capture"
        if not include_in_progress and stage not in HITL_TERMINAL_STAGES:
            continue
        case = _build_hitl_case(sid, state)
        if flagged_only and not case["flagged"]:
            continue
        cases.append(case)

    cases.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    return {
        "count": len(cases),
        "cases": cases,
    }


@app.get("/hitl/summary")
async def get_hitl_summary(include_in_progress: bool = True):
    all_states = await _collect_all_session_states()
    rows = []
    for sid, state in all_states.items():
        stage = state.get("stage") or "data_capture"
        if not include_in_progress and stage not in HITL_TERMINAL_STAGES:
            continue
        rows.append(_build_hitl_case(sid, state))

    total = len(rows)
    flagged = len([r for r in rows if r["flagged"]])
    normal = len([r for r in rows if r["stage"] in {"otp_verification", "complete"}])
    in_progress = len([r for r in rows if r["stage"] not in HITL_TERMINAL_STAGES])
    completed = len([r for r in rows if r["stage"] == "complete"])
    avg_risk = 0
    risk_values = [r["fraud_risk_score"] for r in rows if isinstance(r.get("fraud_risk_score"), int)]
    if risk_values:
        avg_risk = round(sum(risk_values) / len(risk_values), 2)

    return {
        "total": total,
        "flagged": flagged,
        "normal": normal,
        "completed": completed,
        "in_progress": in_progress,
        "avg_risk": avg_risk,
    }


@app.get("/hitl/cases/{session_id}/details")
async def get_hitl_case_details(session_id: str, db: Session = Depends(get_db)):
    from models.compliance_logs import ComplianceStageAlert, LLMDecisionLog
    from audit_logger import AuditLogger

    state = await _ensure_session_state(session_id)
    case = _build_hitl_case(session_id, state)

    alert_rows = (
        db.query(ComplianceStageAlert)
        .filter(ComplianceStageAlert.session_id == session_id)
        .order_by(ComplianceStageAlert.created_at.asc())
        .all()
    )

    alerts = [
        {
            "stage": row.stage,
            "alert_count": int(row.alert_count or 0),
            "alert_summary": row.alert_summary,
            "alert_details": row.alert_details_json or {},
            "overall_flagged": bool(row.overall_flagged),
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "updated_at": row.updated_at.isoformat() if row.updated_at else None,
        }
        for row in alert_rows
    ]

    if not alerts:
        alerts.append(
            {
                "stage": case.get("stage"),
                "alert_count": int(case.get("alerts") or 0),
                "alert_summary": case.get("report") or case.get("obligation"),
                "alert_details": {
                    "fraud_signals": case.get("fraud_signals") or [],
                    "decision_action": case.get("decision_action"),
                    "decision_reason": case.get("decision_reason"),
                },
                "overall_flagged": bool(case.get("flagged")),
                "created_at": case.get("updated_at"),
                "updated_at": case.get("updated_at"),
            }
        )

    audit_session_id = str(case.get("audit_session_id") or "")

    audit_rows = (
        db.query(LLMDecisionLog)
        .filter(
            (LLMDecisionLog.session_id == session_id)
            | (LLMDecisionLog.session_id == audit_session_id)
            | (LLMDecisionLog.audit_session_id == audit_session_id)
        )
        .order_by(LLMDecisionLog.created_at.asc())
        .all()
    )

    if not audit_rows:
        _upsert_llm_decision_summary_log(state, db)
        audit_rows = (
            db.query(LLMDecisionLog)
            .filter(
                (LLMDecisionLog.session_id == session_id)
                | (LLMDecisionLog.session_id == audit_session_id)
                | (LLMDecisionLog.audit_session_id == audit_session_id)
            )
            .order_by(LLMDecisionLog.created_at.asc())
            .all()
        )

    if not audit_rows and audit_session_id:
        log_dir = Path(os.getenv("AUDIT_LOG_DIR", "logs/audit"))
        safe_audit = "".join(c for c in audit_session_id if c.isalnum() or c == "-")
        log_file = log_dir / f"{safe_audit}.jsonl"
        if log_file.exists():
            backfill_logger = AuditLogger(log_dir=log_dir)
            try:
                with open(log_file, "r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        event = json.loads(line)
                        meta = event.get("metadata") if isinstance(event.get("metadata"), dict) else {}
                        meta.setdefault("audit_session_id", audit_session_id)
                        backfill_logger._write_db_log(
                            session_id=event.get("session_id") or audit_session_id,
                            event_type=event.get("event_type") or "unknown_event",
                            input_data=event.get("input_data") if isinstance(event.get("input_data"), dict) else None,
                            output_data=event.get("output_data") if isinstance(event.get("output_data"), dict) else None,
                            decision=event.get("decision"),
                            metadata=meta,
                        )
            except Exception as exc:
                print(f"[DEBUG][hitl] audit_backfill_failed sid={session_id} err={exc}")

            audit_rows = (
                db.query(LLMDecisionLog)
                .filter(
                    (LLMDecisionLog.session_id == session_id)
                    | (LLMDecisionLog.session_id == audit_session_id)
                    | (LLMDecisionLog.audit_session_id == audit_session_id)
                )
                .order_by(LLMDecisionLog.created_at.asc())
                .all()
            )

    audit_logs = [
        {
            "id": row.id,
            "stage": row.stage,
            "event_type": row.event_type,
            "decision_source": row.decision_source,
            "decision": row.decision,
            "friendly_text": row.friendly_text,
            "log_hash": row.log_hash,
            "input_payload": row.input_payload_json,
            "output_payload": row.output_payload_json,
            "metadata": row.metadata_json,
            "created_at": row.created_at.isoformat() if row.created_at else None,
        }
        for row in audit_rows
    ]

    overall_alerts = sum(item["alert_count"] for item in alerts)
    total_flagged_stages = len([item for item in alerts if item["overall_flagged"]])

    aml_raw = state.get("aml_raw_results") or {}
    aml_checks: list[dict] = []

    rbi = aml_raw.get("rbi") or {}
    if rbi:
        aml_checks.append(
            {
                "check": "RBI Caution List",
                "status": "hit" if rbi.get("hit") else "clear",
                "detail": rbi.get("reason") or "No caution match",
                "important": bool(rbi.get("hit")),
            }
        )

    ofac = aml_raw.get("ofac") or {}
    if ofac:
        aml_checks.append(
            {
                "check": "OFAC Sanctions",
                "status": "hit" if ofac.get("hit") else ("near_miss" if ofac.get("near_miss") else "clear"),
                "detail": (
                    f"{ofac.get('matched_name') or 'No match'}"
                    + (f" (score {ofac.get('match_score')})" if ofac.get("match_score") is not None else "")
                ),
                "important": bool(ofac.get("hit") or ofac.get("near_miss")),
            }
        )

    pep = aml_raw.get("pep") or {}
    if pep:
        aml_checks.append(
            {
                "check": "PEP Screening",
                "status": "hit" if pep.get("hit") else ("near_miss" if pep.get("near_miss") else "clear"),
                "detail": pep.get("position") or pep.get("matched_name") or "No PEP match",
                "important": bool(pep.get("hit") or pep.get("near_miss")),
            }
        )

    rules = (aml_raw.get("rules") or {}).get("triggered_rules") or []
    if rules:
        aml_checks.append(
            {
                "check": "Risk Rules",
                "status": "triggered",
                "detail": ", ".join(str(rule.get("rule_id")) for rule in rules),
                "important": True,
            }
        )

    return {
        "case": case,
        "alerts_by_stage": alerts,
        "overall": {
            "total_alerts": overall_alerts,
            "flagged_stages": total_flagged_stages,
            "total_stages": len(alerts),
        },
        "audit_logs": audit_logs,
        "aml_checks": aml_checks,
    }


async def _post_to_accuverify(
    path: str,
    *,
    params: dict | None = None,
    files: dict | None = None,
    timeout: httpx.Timeout | None = None,
) -> dict:
    """Forward to AccuVerify; map connection errors to 503 instead of ASGI 500."""
    client = get_http_client()
    url = f"{ACCUVERIFY_URL}/{path.lstrip('/')}"
    try:
        resp = await client.post(url, params=params, files=files, timeout=timeout)
    except httpx.ConnectError as exc:
        logger.warning("AccuVerify unreachable at %s (%s)", url, exc)
        raise HTTPException(
            status_code=503,
            detail={
                "verified": False,
                "error": "accuverify_unreachable",
                "message": (
                    f"Cannot connect to AccuVerify at {ACCUVERIFY_URL}. "
                    "Start the Verify service, e.g.: "
                    "cd AccuEntry_Verify && python -m uvicorn main:app --host 127.0.0.1 --port 9000"
                ),
            },
        ) from exc
    except httpx.TimeoutException as exc:
        logger.warning("AccuVerify timeout %s", url)
        raise HTTPException(
            status_code=504,
            detail={
                "verified": False,
                "error": "accuverify_timeout",
                "message": "AccuVerify did not respond in time.",
            },
        ) from exc

    try:
        return resp.json()
    except ValueError:
        raise HTTPException(
            status_code=502,
            detail={
                "verified": False,
                "error": "bad_gateway",
                "message": f"AccuVerify returned non-JSON (HTTP {resp.status_code}).",
            },
        )


@app.post("/kyc/pan")
async def proxy_pan(session_id: str = Form(...), file: UploadFile = File(...)):
    return await _post_to_accuverify(
        "upload-pan",
        params={"user_id": session_id},
        files={"file": (file.filename, await file.read(), file.content_type)},
    )


@app.post("/kyc/aadhaar")
async def proxy_aadhaar(session_id: str = Form(...), file: UploadFile = File(...)):
    return await _post_to_accuverify(
        "upload-aadhaar",
        params={"user_id": session_id},
        files={"file": (file.filename, await file.read(), file.content_type)},
    )


@app.post("/kyc/selfie")
async def proxy_selfie(session_id: str = Form(...), file: UploadFile = File(...)):
    return await _post_to_accuverify(
        "upload-selfie",
        params={"user_id": session_id},
        files={"file": (file.filename, await file.read(), file.content_type)},
        timeout=httpx.Timeout(60.0),
    )


@app.post("/kyc/video-kyc")
async def proxy_video_kyc(session_id: str = Form(...), file: UploadFile = File(...)):
    return await _post_to_accuverify(
        "upload-video-kyc",
        params={"user_id": session_id},
        files={"file": (file.filename, await file.read(), file.content_type)},
        timeout=httpx.Timeout(120.0),
    )


@app.post("/kyc/approve")
async def proxy_approve(session_id: str = Form(...)):
    return await _post_to_accuverify(
        "agent/approve-kyc",
        params={"user_id": session_id, "agent_id": "accuentry-bot"},
    )


@app.on_event("shutdown")
async def _shutdown_http_pool() -> None:
    await close_http_client()
