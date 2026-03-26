import os
import asyncio
from fastapi import FastAPI, Form, UploadFile, File
import httpx
from fastapi.middleware.cors import CORSMiddleware
from schemas.chat import ChatRequest, ChatResponse
from supervisor import onboarding_graph
from state import OnboardingState
from core.database import engine, Base
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

print(f"Using AccuVerify URL: {ACCUVERIFY_URL}")
def _initial_onboarding_state(session_id: str) -> OnboardingState:
    return {
        "session_id": session_id,
        "messages": [],
        "stage": "data_capture",
        "full_name": None,
        "dob": None,
        "pan_number": None,
        "address": None,
        "account_type": None,
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

        updated: OnboardingState = {
            **latest,
            "aml_raw_results": result.get("aml_raw_results", latest.get("aml_raw_results")),
            "aml_risk_score": result.get("aml_risk_score", latest.get("aml_risk_score")),
            "aml_status": result.get("aml_status", latest.get("aml_status", "pending")),
            "stage": result.get("stage", latest.get("stage")),  # ← FIX: propagate stage from AML result
            "messages": result.get("messages", latest.get("messages", [])),  # ← FIX: propagate messages
            "aml_in_background": False,
            "aml_completed": result.get("aml_status") in ("clear", "flagged"),
        }

        sessions[session_id] = updated
    except Exception as exc:
        latest = sessions.get(session_id)
        if latest:
            sessions[session_id] = {
                **latest,
                "aml_in_background": False,
                "aml_status": "pending",
            }
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
        "manual_review": (4, "Manual Review"),
        "complete": (5, "Account Activated"),
        "rejected": (1, "Application rejected"),
    }
    return mapping.get(stage, (1, "Detail Capture"))


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        sid = request.session_id
        if sid not in sessions:
            sessions[sid] = _initial_onboarding_state(sid)

        state: OnboardingState = {**sessions[sid], "messages": list(sessions[sid]["messages"])}
        text = (request.user_input or "").strip()
        if text:
            state["messages"].append({"role": "user", "text": text})

        new_state = await onboarding_graph.ainvoke(
            state,
            config={"recursion_limit": 50},
        )
        sessions[sid] = new_state
        _start_aml_if_needed(sid)

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
        )
    except Exception as e:
        print(f"Error handling chat: {e}")
        raise e


@app.post("/kyc/pan")
async def proxy_pan(session_id: str = Form(...), file: UploadFile = File(...)):
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{ACCUVERIFY_URL}/upload-pan",
            params={"user_id": session_id},
            files={"file": (file.filename, await file.read(), file.content_type)},
        )
    return resp.json()


@app.post("/kyc/aadhaar")
async def proxy_aadhaar(session_id: str = Form(...), file: UploadFile = File(...)):
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{ACCUVERIFY_URL}/upload-aadhaar",
            params={"user_id": session_id},
            files={"file": (file.filename, await file.read(), file.content_type)},
        )
    return resp.json()


@app.post("/kyc/selfie")
async def proxy_selfie(session_id: str = Form(...), file: UploadFile = File(...)):
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
        resp = await client.post(
            f"{ACCUVERIFY_URL}/upload-selfie",
            params={"user_id": session_id},
            files={"file": (file.filename, await file.read(), file.content_type)},
        )
    return resp.json()


@app.post("/kyc/approve")
async def proxy_approve(session_id: str = Form(...)):
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{ACCUVERIFY_URL}/agent/approve-kyc",
            params={"user_id": session_id, "agent_id": "accuentry-bot"},
        )
    return resp.json()
