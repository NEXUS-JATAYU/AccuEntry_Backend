"""
Agent invocation: local in-process LangGraph or remote AccuEntry_Agents microservice.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

from core.security import AGENT_SERVICE_API_KEY
from state import OnboardingState

logger = logging.getLogger(__name__)

AGENT_SERVICE_URL = os.getenv("AGENT_SERVICE_URL", "").strip().rstrip("/")
USE_AGENT_MICROSERVICE = os.getenv("USE_AGENT_MICROSERVICE", "false").lower() in {
    "1",
    "true",
    "yes",
} or bool(AGENT_SERVICE_URL)

_RECURSION_LIMIT = int(os.getenv("AGENT_RECURSION_LIMIT", "50"))


def _use_remote() -> bool:
    return USE_AGENT_MICROSERVICE and bool(AGENT_SERVICE_URL)


async def _remote_invoke(endpoint: str, state: OnboardingState) -> dict[str, Any]:
    url = f"{AGENT_SERVICE_URL}{endpoint}"
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if AGENT_SERVICE_API_KEY:
        headers["X-Agent-Service-Key"] = AGENT_SERVICE_API_KEY

    payload = {"state": _state_to_json(state)}

    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
        resp = await client.post(url, json=payload, headers=headers)
        if resp.status_code >= 400:
            logger.error("agent_service_error url=%s status=%s body=%s", url, resp.status_code, resp.text[:500])
            resp.raise_for_status()
        data = resp.json()
    return data.get("state") or data


def _state_to_json(state: OnboardingState) -> dict[str, Any]:
    """Serialize state for HTTP transport (messages must be JSON-safe)."""
    out: dict[str, Any] = dict(state)
    messages = out.get("messages") or []
    if isinstance(messages, list):
        out["messages"] = [
            m if isinstance(m, dict) else {"role": "user", "text": str(m)}
            for m in messages
        ]
    return out


async def invoke_onboarding_graph(state: OnboardingState) -> OnboardingState:
    if _use_remote():
        result = await _remote_invoke("/agents/onboarding/invoke", state)
        return {**state, **result}

    from supervisor import onboarding_graph

    return await onboarding_graph.ainvoke(state, config={"recursion_limit": _RECURSION_LIMIT})


async def invoke_aml_graph(state: OnboardingState) -> OnboardingState:
    if _use_remote():
        result = await _remote_invoke("/agents/aml/invoke", state)
        return {**state, **result}

    from agents.aml.aml_screening import build_aml_graph

    aml_graph = build_aml_graph()
    return await aml_graph.ainvoke(state, config={"recursion_limit": _RECURSION_LIMIT})


async def invoke_faq_node(state: OnboardingState) -> dict[str, Any]:
    if _use_remote():
        return await _remote_invoke("/agents/faq/invoke", state)

    from agents.faq.faq_agent import faq_node

    return await faq_node(state)


def agent_runtime_mode() -> str:
    return "microservice" if _use_remote() else "embedded"
