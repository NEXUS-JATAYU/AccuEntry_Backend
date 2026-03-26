"""
KYC approval subgraph: calls AccuVerify agent approve-kyc endpoint.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from core.http_client_pool import get_http_client
from state import OnboardingState

ACCUVERIFY_URL = os.getenv("ACCUVERIFY_URL", "http://localhost:8001").rstrip("/")


async def check_node(state: OnboardingState) -> dict[str, Any]:
    url = f"{ACCUVERIFY_URL.rstrip('/')}/agent/approve-kyc"
    try:
        client = get_http_client()
        resp = await client.post(
            url,
            params={
                "user_id": state["session_id"],
                "agent_id": "accuentry-bot",
            },
        )
        resp.raise_for_status()
        body = resp.json()
    except (httpx.HTTPError, ValueError):
        return {
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

    status = body.get("status")
    if status == "kyc_approved":
        return {
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
