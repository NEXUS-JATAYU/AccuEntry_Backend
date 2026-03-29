"""
Document verification subgraph: prompt uploads, poll AccuVerify KYC status, handle failures.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from core.http_client_pool import get_http_client
from state import OnboardingState

ACCUVERIFY_URL = os.getenv("ACCUVERIFY_URL", "http://localhost:9000")


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
    except (httpx.HTTPError, ValueError):
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
        }
    )

    if pan_v and aadhaar_v and face_v and video_kyc_v:
        base.update(
            {
                "stage": "kyc_approval",
                "progress": 55,
                "requires_upload": False,
                "messages": [
                    {
                        "role": "assistant",
                        "text": (
                            "Live kyc verified. Your application is moving to "
                            "KYC review."
                        ),
                    }
                ],
            }
        )
        return base
        
    if pan_v and aadhaar_v and face_v and not video_kyc_v:
        import json
        payload_str = json.dumps({
            "type": "LIVE_KYC_REQUESTED", 
            "payload": {"message": "click on this link to start live KYC video"}
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
        base["doc_failure_type"] = "pan"
        return base
    if aadhaar_failed:
        base["doc_failure_type"] = "aadhaar"
        return base
    if face_failed:
        base["doc_failure_type"] = "face"
        return base
    if video_kyc_failed:
        base["doc_failure_type"] = "video_kyc"
        return base

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
