"""
KYC approval subgraph: calls AccuVerify agent approve-kyc endpoint.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from core.http_client_pool import get_http_client
from core.verify_client import get_verify_request_headers
from state import OnboardingState

logger = logging.getLogger(__name__)


def _accuverify_base_url() -> str:
    """Read at call time so .env is loaded before first graph run; match doc_verify/main defaults."""
    return os.getenv("ACCUVERIFY_URL", "http://127.0.0.1:9000").rstrip("/")


async def check_node(state: OnboardingState) -> dict[str, Any]:
    base_kyc: dict[str, Any] = {
        "kyc_data": state.get("kyc_data") or {},
        "pan_verified": bool(state.get("pan_verified")),
        "aadhaar_verified": bool(state.get("aadhaar_verified")),
        "face_verified": bool(state.get("face_verified")),
    }
    url = f"{_accuverify_base_url()}/agent/approve-kyc"
    try:
        client = get_http_client()
        resp = await client.post(
            url,
            params={
                "user_id": state["session_id"],
                "agent_id": "accuentry-bot",
            },
            headers=get_verify_request_headers(),
        )
        resp.raise_for_status()
        body = resp.json()
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("approve-kyc request failed url=%s err=%s", url, exc)
        return {
            **base_kyc,
            "stage": "rejected",
            "kyc_status": "rejected",
            "messages": [
                {
                    "role": "assistant",
                    "text": (
                        "We could not complete KYC review with the verification service. "
                        "Please try again later."
                    ),
                }
            ],
        }

    if body.get("error") == "kyc_record_not_found":
        return {
            **base_kyc,
            "stage": "doc_verification",
            "kyc_status": None,
            "messages": [
                {
                    "role": "assistant",
                    "text": (
                        "We have no verification record for your uploads yet. "
                        "Please finish PAN, Aadhaar, and selfie steps, then try again."
                    ),
                }
            ],
        }

    status = body.get("status")
    if status == "kyc_approved":
        kyc_data = body.get("kyc_data") or state.get("kyc_data") or {}
        return {
            "kyc_data": kyc_data,
            "pan_verified": bool(body.get("pan_verified", state.get("pan_verified"))),
            "aadhaar_verified": bool(body.get("aadhaar_verified", state.get("aadhaar_verified"))),
            "face_verified": bool(body.get("face_verified", state.get("face_verified"))),
            "kyc_status": "approved",
            "stage": "fraud_check",
            "progress": 65,
            "messages": [
                {
                    "role": "assistant",
                    "text": "KYC approved. Continuing to final checks while AML screening runs in the background.",
                }
            ],
        }

    if status == "cannot_approve":
        reason = body.get("message") or "Automatic verification is not complete."
        return {
            **base_kyc,
            "stage": "rejected",
            "kyc_status": "rejected",
            "messages": [
                {
                    "role": "assistant",
                    "text": f"We cannot approve your KYC at this time: {reason}",
                }
            ],
        }

    return {
        **base_kyc,
        "stage": "rejected",
        "kyc_status": "rejected",
        "messages": [
            {
                "role": "assistant",
                "text": "We could not determine your KYC outcome. Please contact support.",
            }
        ],
    }


def build_kyc_approval_graph() -> CompiledStateGraph:
    workflow = StateGraph(OnboardingState)
    workflow.add_node("check_node", check_node)
    workflow.add_edge(START, "check_node")
    workflow.add_edge("check_node", END)
    return workflow.compile()
