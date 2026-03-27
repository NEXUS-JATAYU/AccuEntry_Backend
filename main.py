import os
import asyncio
from fastapi import FastAPI, Form, UploadFile, File, Depends
import httpx
from fastapi.middleware.cors import CORSMiddleware
from schemas.chat import ChatRequest, ChatResponse
from supervisor import onboarding_graph
from state import OnboardingState
from sqlalchemy.orm import Session
from core.database import engine, Base, get_db
from core.http_client_pool import close_http_client, get_http_client
from agents.aml.aml_screening import build_aml_graph
import models.customer_info
from scripts.init_aml_indices import init_aml_indices

# Initialize DB tables
Base.metadata.create_all(bind=engine)

# Initialize AML MongoDB indices
init_aml_indices()

app = FastAPI()
ACCUVERIFY_URL = os.getenv("ACCUVERIFY_URL", "http://127.0.0.1:8001").rstrip("/")
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
MAX_SESSION_MESSAGES = int(os.getenv("MAX_SESSION_MESSAGES", "120"))

print(f"Using AccuVerify URL: {ACCUVERIFY_URL}")
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
        "video_kyc_status": None,
        "risk_model_label": None,
        "risk_model_confidence": None,
        "kyc_data": None,
    }


def _should_start_aml(state: OnboardingState) -> bool:
    if state.get("kyc_status") != "approved":
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
        }

        sessions[session_id] = _trim_messages(updated)
    except Exception as exc:
        latest = sessions.get(session_id)
        if latest:
            sessions[session_id] = _trim_messages({
                **latest,
                "aml_in_background": False,
                "aml_status": "pending",
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
    }
    aml_tasks[session_id] = asyncio.create_task(_run_aml_background(session_id))


def _last_assistant_text(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "assistant":
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
        "rejected": (1, "Application rejected"),
    }
    return mapping.get(stage, (1, "Detail Capture"))

def _save_state_to_db(state: OnboardingState, db: Session) -> None:
    from models.customer_info import CustomerDetails
    from datetime import datetime
    
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


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, db: Session = Depends(get_db)):
    try:
        sid = request.session_id
        if sid not in sessions:
            sessions[sid] = _initial_onboarding_state(sid)

        state: OnboardingState = {**sessions[sid], "messages": list(sessions[sid]["messages"])}
        text = (request.user_input or "").strip()

        # ── OTP verification intercept (no LLM needed) ──────────
        if state["stage"] == "otp_verification" and text:
            import re as _re
            import json as _json
            import uuid as _uuid
            from datetime import datetime, timezone
            from agents.decision.otp_service import (
                verify_otp, send_confirmation_email, mask_email, is_otp_locked,
            )

            if text:
                state["messages"].append({"role": "user", "text": text})

            digits = _re.sub(r"\D", "", text)
            otp_session_id = state.get("audit_session_id") or sid

            if is_otp_locked(otp_session_id):
                reply = "Too many incorrect attempts. Please restart the activation process or contact support."
                state["messages"].append({"role": "assistant", "text": reply})
                sessions[sid] = _trim_messages(state)
                stage = state["stage"]
                step, main_step = _ui_step_and_label(stage)
                return ChatResponse(
                    message=reply, progress=state["progress"],
                    stage=stage, step=step, current_main_step=main_step,
                    otp_required=True,
                )

            if len(digits) != 4:
                reply = "Please enter a valid 4-digit activation code."
                state["messages"].append({"role": "assistant", "text": reply})
                sessions[sid] = _trim_messages(state)
                stage = state["stage"]
                step, main_step = _ui_step_and_label(stage)
                return ChatResponse(
                    message=reply, progress=state["progress"],
                    stage=stage, step=step, current_main_step=main_step,
                    otp_required=True,
                )

            success, msg = verify_otp(otp_session_id, digits)

            if not success:
                state["messages"].append({"role": "assistant", "text": msg})
                sessions[sid] = _trim_messages(state)
                stage = state["stage"]
                step, main_step = _ui_step_and_label(stage)
                return ChatResponse(
                    message=msg, progress=state["progress"],
                    stage=stage, step=step, current_main_step=main_step,
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

            state["stage"] = "complete"
            state["progress"] = 100
            state["messages"].append({"role": "assistant", "text": activation_msg})
            sessions[sid] = _trim_messages(state)
            _save_state_to_db(sessions[sid], db)

            step, main_step = _ui_step_and_label("complete")
            return ChatResponse(
                message=activation_msg, progress=100,
                stage="complete", completed=True,
                step=step, current_main_step=main_step,
                otp_required=False,
            )

        # ── Normal graph flow ───────────────────────────────────
        if text:
            state["messages"].append({"role": "user", "text": text})

        new_state = await onboarding_graph.ainvoke(
            state,
            config={"recursion_limit": 50},
        )
        sessions[sid] = _trim_messages(new_state)
        _start_aml_if_needed(sid)
        _save_state_to_db(sessions[sid], db)

        latest_state = sessions[sid]

        stage = latest_state["stage"]
        step, main_step = _ui_step_and_label(stage)
        raw_reply = _last_assistant_text(latest_state["messages"])
        reply = raw_reply or _fallback_assistant_message(
            stage,
            latest_state["requires_upload"],
        )
        return ChatResponse(
            message=reply,
            progress=latest_state["progress"],
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


@app.post("/kyc/pan")
async def proxy_pan(session_id: str = Form(...), file: UploadFile = File(...)):
    client = get_http_client()
    resp = await client.post(
        f"{ACCUVERIFY_URL}/upload-pan",
        params={"user_id": session_id},
        files={"file": (file.filename, await file.read(), file.content_type)},
    )
    return resp.json()


@app.post("/kyc/aadhaar")
async def proxy_aadhaar(session_id: str = Form(...), file: UploadFile = File(...)):
    client = get_http_client()
    resp = await client.post(
        f"{ACCUVERIFY_URL}/upload-aadhaar",
        params={"user_id": session_id},
        files={"file": (file.filename, await file.read(), file.content_type)},
    )
    return resp.json()


@app.post("/kyc/selfie")
async def proxy_selfie(session_id: str = Form(...), file: UploadFile = File(...)):
    client = get_http_client()
    resp = await client.post(
        f"{ACCUVERIFY_URL}/upload-selfie",
        params={"user_id": session_id},
        files={"file": (file.filename, await file.read(), file.content_type)},
        timeout=httpx.Timeout(60.0),
    )
    return resp.json()


@app.post("/kyc/approve")
async def proxy_approve(session_id: str = Form(...)):
    client = get_http_client()
    resp = await client.post(
        f"{ACCUVERIFY_URL}/agent/approve-kyc",
        params={"user_id": session_id, "agent_id": "accuentry-bot"},
    )
    return resp.json()


@app.on_event("shutdown")
async def _shutdown_http_pool() -> None:
    await close_http_client()
