"""
Document verification subgraph: prompt uploads, poll AccuVerify KYC status, handle failures.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from rag_service import retrieve_as_context
from core.http_client_pool import get_http_client
from memory_manager import AgentMemoryManager
from state import OnboardingState

ACCUVERIFY_URL = os.getenv("ACCUVERIFY_URL", "http://localhost:9000")
_memory = AgentMemoryManager()
logger = logging.getLogger(__name__)


def _store_doc_verify_memory(
    state: OnboardingState,
    *,
    status: str,
    outcome_stage: str,
    verify_payload: dict[str, Any],
) -> None:
    session_id = state.get("audit_session_id") or state.get("session_id") or "unknown"
    _memory.store_interaction(
        session_id=session_id,
        agent_name="doc_verify",
        input_data={
            "session_id": state.get("session_id"),
            "doc_failure_type": state.get("doc_failure_type"),
            "pan_verified": state.get("pan_verified"),
            "aadhaar_verified": state.get("aadhaar_verified"),
            "face_verified": state.get("face_verified"),
            "verify_payload": verify_payload,
        },
        output_data={
            "status": status,
            "outcome_stage": outcome_stage,
        },
        decision=status,
        metadata={
            "audit_session_id": state.get("audit_session_id") or session_id,
            "workflow_stage": state.get("stage") or "doc_verification",
            "outcome_stage": outcome_stage,
            "doc_failure_type": state.get("doc_failure_type") or "",
        },
        event_type="doc_verify_outcome",
    )


def request_uploads_node(state: OnboardingState) -> dict[str, Any]:
    if state.get("progress", 0) >= 40 and state.get("requires_upload"):
        # No merge: upload prompt already shown; avoids duplicate assistant messages.
        return {}
    return {
        "requires_upload": True,
        "progress": 40,
        "messages": [
            {
                "role": "assistant",
                "text": (
                    "Please upload clear photos of your PAN card, your Aadhaar card, "
                    "and a selfie. Use the upload area when it appears — PAN first, "
                    "then Aadhaar, then selfie for face matching."
                ),
            }
        ],
    }


async def poll_status_node(state: OnboardingState) -> dict[str, Any]:
    base: dict[str, Any] = {
        "doc_failure_type": None,
        "pan_verified": bool(state.get("pan_verified")),
        "aadhaar_verified": bool(state.get("aadhaar_verified")),
        "face_verified": bool(state.get("face_verified")),
        "video_kyc_status": state.get("video_kyc_status") or "pending",
    }
    url = f"{ACCUVERIFY_URL.rstrip('/')}/kyc/status"
    try:
        client = get_http_client()
        resp = await client.get(url, params={"user_id": state["session_id"]})
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPError as exc:
        logger.warning(
            "AccuVerify kyc/status unreachable session=%s url=%s err=%s",
            state.get("session_id"),
            url,
            exc,
        )
        _store_doc_verify_memory(
            state,
            status="verify_service_unavailable",
            outcome_stage=state.get("stage") or "doc_verification",
            verify_payload={"error": str(exc)},
        )
        base["messages"] = [
            {
                "role": "assistant",
                "text": (
                    "We could not reach the verification service. "
                    "Please try again in a moment."
                ),
            }
        ]
        return base
    except ValueError as exc:
        logger.warning(
            "AccuVerify kyc/status invalid JSON session=%s err=%s",
            state.get("session_id"),
            exc,
        )
        base["messages"] = [
            {
                "role": "assistant",
                "text": (
                    "We could not read the verification status. "
                    "Please try again in a moment."
                ),
            }
        ]
        return base

    if not isinstance(data, dict):
        logger.warning("AccuVerify kyc/status unexpected payload session=%s", state.get("session_id"))
        data = {}

    pan_v = bool(data.get("pan_verified"))
    aadhaar_v = bool(data.get("aadhaar_verified"))
    face_v = bool(data.get("face_verified"))
    video_kyc_v = bool(data.get("video_kyc_verified"))
    pan_failed = bool(data.get("pan_failed"))
    aadhaar_failed = bool(data.get("aadhaar_failed"))
    face_failed = bool(data.get("face_failed"))
    video_kyc_failed = bool(data.get("video_kyc_failed"))

    base.update(
        {
            "pan_verified": pan_v,
            "aadhaar_verified": aadhaar_v,
            "face_verified": face_v,
            "video_kyc_status": "verified" if video_kyc_v else ("failed" if video_kyc_failed else "pending"),
            "document_verified": pan_v and aadhaar_v and face_v,
        }
    )
    for doc_key in ("document_name", "document_dob", "document_address"):
        if doc_key in data:
            base[doc_key] = data.get(doc_key)

    if pan_v and aadhaar_v and face_v and video_kyc_v:
        _store_doc_verify_memory(
            state,
            status="all_verified",
            outcome_stage="kyc_approval",
            verify_payload={
                "pan_verified": pan_v,
                "aadhaar_verified": aadhaar_v,
                "face_verified": face_v,
                "video_kyc_verified": video_kyc_v,
                "pan_confidence": data.get("pan_confidence"),
                "aadhaar_confidence": data.get("aadhaar_confidence"),
                "face_confidence": data.get("face_confidence"),
                "mismatch_types": data.get("mismatch_types") or [],
                "doc_quality_signals": data.get("doc_quality_signals") or [],
            },
        )
        base.update(
            {
                "stage": "kyc_approval",
                "progress": 55,
                "requires_upload": False,
                "messages": [
                    {
                        "role": "assistant",
                        "text": (
                            "Live KYC verified. Your application is moving to "
                            "KYC review."
                        ),
                    }
                ],
            }
        )
        return base

    if pan_v and aadhaar_v and face_v and not video_kyc_v:
        _store_doc_verify_memory(
            state,
            status="live_kyc_required",
            outcome_stage="doc_verification",
            verify_payload={
                "pan_verified": pan_v,
                "aadhaar_verified": aadhaar_v,
                "face_verified": face_v,
                "video_kyc_verified": video_kyc_v,
                "video_kyc_failed": video_kyc_failed,
            },
        )
        import json
        payload_str = json.dumps({
            "type": "LIVE_KYC_REQUESTED", 
            "payload": {
                "message": (
                    "Your PAN, Aadhaar, and selfie are verified.\n"
                    "Please click the button below to start Live KYC video verification."
                )
            }
        })
        last_msgs = state.get("messages", [])
        already_prompted = False
        for m in reversed(last_msgs):
            if m.get("role") == "user":
                break
            if m.get("role") == "assistant" and ("LIVE_KYC_REQUESTED" in m.get("text", "")):
                already_prompted = True
                break
                
        if not already_prompted:
            base.update({
                "stage": "doc_verification",
                "progress": max(int(state.get("progress", 40)), 50),
                "requires_upload": False,
                "messages": [
                    {
                        "role": "assistant",
                        "text": payload_str
                    }
                ]
            })
            return base

    if pan_failed:
        _store_doc_verify_memory(
            state,
            status="pan_failed",
            outcome_stage="doc_verification",
            verify_payload={
                "pan_failed": True,
                "pan_confidence": data.get("pan_confidence"),
                "mismatch_types": data.get("mismatch_types") or [],
                "doc_quality_signals": data.get("doc_quality_signals") or [],
            },
        )
        base["doc_failure_type"] = "pan"
        return base
    if aadhaar_failed:
        _store_doc_verify_memory(
            state,
            status="aadhaar_failed",
            outcome_stage="doc_verification",
            verify_payload={
                "aadhaar_failed": True,
                "aadhaar_confidence": data.get("aadhaar_confidence"),
                "mismatch_types": data.get("mismatch_types") or [],
                "doc_quality_signals": data.get("doc_quality_signals") or [],
            },
        )
        base["doc_failure_type"] = "aadhaar"
        return base
    if face_failed:
        _store_doc_verify_memory(
            state,
            status="face_failed",
            outcome_stage="doc_verification",
            verify_payload={
                "face_failed": True,
                "face_confidence": data.get("face_confidence"),
                "mismatch_types": data.get("mismatch_types") or [],
                "doc_quality_signals": data.get("doc_quality_signals") or [],
            },
        )
        base["doc_failure_type"] = "face"
        return base
    if video_kyc_failed:
        base["doc_failure_type"] = "video_kyc"
        return base

    _store_doc_verify_memory(
        state,
        status="verification_pending",
        outcome_stage=state.get("stage") or "doc_verification",
        verify_payload={
            "pan_verified": pan_v,
            "aadhaar_verified": aadhaar_v,
            "face_verified": face_v,
            "pan_confidence": data.get("pan_confidence"),
            "aadhaar_confidence": data.get("aadhaar_confidence"),
            "face_confidence": data.get("face_confidence"),
            "mismatch_types": data.get("mismatch_types") or [],
            "doc_quality_signals": data.get("doc_quality_signals") or [],
        },
    )
    base["messages"] = [
        {
            "role": "assistant",
            "text": (
                "We are still verifying your uploads. "
                "This may take a few moments — you can send another message to check status."
            ),
        }
    ]
    return base


def failure_node(state: OnboardingState) -> dict[str, Any]:
    kind = state.get("doc_failure_type") or "pan"
    rag_query = f"valid documents for {kind} proof" if kind != "face" \
                else "selfie face matching requirements"
    policy_excerpt = retrieve_as_context(rag_query, top_k=2)
    _store_doc_verify_memory(
        state,
        status=f"retry_requested_{kind}",
        outcome_stage="doc_verification",
        verify_payload={"failure_kind": kind, "policy_excerpt": policy_excerpt},
    )
    import json
    if kind == "video_kyc":
        msg = json.dumps({
            "type": "LIVE_KYC_REQUESTED",
            "payload": {
                "message": "Your Live KYC video could not be verified — please click on this link to start live KYC video again."
            }
        })
        return {
            "doc_failure_type": None,
            "requires_upload": False,
            "messages": [{"role": "assistant", "text": msg}],
        }
        
    texts = {
        "pan": "Your PAN could not be verified — please re-upload a clearer image.",
        "aadhaar": "Your Aadhaar could not be verified — please re-upload a clearer image.",
        "face": (
            "Your selfie could not be matched to your Aadhaar photo — "
            "please re-upload a clearer selfie."
        ),
    }
    msg = texts.get(kind, texts["pan"])
    _store_doc_verify_memory(
        state,
        status=f"retry_requested_{kind}",
        outcome_stage="doc_verification",
        verify_payload={"failure_kind": kind},
    )
    return {
        "doc_failure_type": None,
        "requires_upload": True,
        "messages": [{"role": "assistant", "text": msg}],
    }


def _route_after_poll(state: OnboardingState) -> Any:
    if state.get("doc_failure_type"):
        return "failure_node"
    return END


def build_doc_verify_graph() -> CompiledStateGraph:
    workflow = StateGraph(OnboardingState)
    workflow.add_node("request_uploads_node", request_uploads_node)
    workflow.add_node("poll_status_node", poll_status_node)
    workflow.add_node("failure_node", failure_node)
    workflow.add_edge(START, "request_uploads_node")
    workflow.add_edge("request_uploads_node", "poll_status_node")
    workflow.add_conditional_edges(
        "poll_status_node",
        _route_after_poll,
        {
            "failure_node": "failure_node",
            END: END,
        },
    )
    workflow.add_edge("failure_node", END)
    return workflow.compile()
