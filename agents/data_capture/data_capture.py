"""
Data capture LangGraph subgraph: collects five onboarding fields into OnboardingState.

Flow per invoke: entry_node → capture_node (LLM) → validate_node → END.
"""

from __future__ import annotations
import os
from typing import Any, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agents.data_capture.data_capture_validator import (
    validate_date,
    validate_name,
    validate_pan,
)
from state import OnboardingState

FIELD_ORDER: tuple[str, ...] = (
    "full_name",
    "dob",
    "pan_number",
    "address",
    "account_type",
)

FIELD_QUESTIONS: dict[str, str] = {
    "full_name": "Enter full name:",
    "dob": "Enter DOB (YYYY-MM-DD):",
    "pan_number": "Enter PAN:",
    "address": "Enter address:",
    "account_type": "Enter account type (Savings/Current):",
}


def _is_set(value: Optional[str]) -> bool:
    return value is not None and bool(str(value).strip())


def _first_missing(state: OnboardingState) -> Optional[str]:
    for key in FIELD_ORDER:
        if not _is_set(state.get(key)):  # type: ignore[arg-type]
            return key
    return None


def _last_user_text(state: OnboardingState) -> str:
    for m in reversed(state.get("messages", [])):
        if m.get("role") == "user":
            return (m.get("text") or "").strip()
    return ""


def _llm() -> ChatOllama:
    model =  "gemma2:2b"
    return ChatOllama(model=model, temperature=0)


_llm_singleton: Optional[ChatOllama] = None


def _get_llm() -> ChatOllama:
    global _llm_singleton
    if _llm_singleton is None:
        _llm_singleton = _llm()
    return _llm_singleton


def entry_node(state: OnboardingState) -> dict[str, Any]:
    target = _first_missing(state)
    out: dict[str, Any] = {"capture_target": target, "capture_error": None}

    if target is None:
        return out

    # Only ask the question if we haven't already asked it
    # (avoid duplicate questions for the same field)
    current_target = state.get("capture_target")
    if current_target == target:
        # We're already asking for this field, don't ask again
        return out

    msgs = state.get("messages", [])
    last_is_user = bool(msgs) and msgs[-1].get("role") == "user"
    if not last_is_user:
        out["messages"] = [{"role": "assistant", "text": FIELD_QUESTIONS[target]}]
    return out


async def capture_node(state: OnboardingState) -> dict[str, Any]:
    target = state.get("capture_target")
    if not target:
        return {"capture_candidate": None}

    user_text = _last_user_text(state)
    if not user_text:
        return {"capture_candidate": None}

    llm = _get_llm()
    system = SystemMessage(
        content=(
            f"You extract a single field for a bank onboarding form. "
            f'Field name: "{target}". '
            "Reply with only the extracted value, no quotes or explanation. "
            'If the field is account_type, reply with exactly "Savings" or "Current" '
            "when possible; otherwise reply with the user\u2019s wording. "
            "If nothing relevant is present, reply with an empty string."
        )
    )
    human = HumanMessage(content=user_text)
    resp = await llm.ainvoke([system, human])
    raw = (getattr(resp, "content", None) or "").strip()
    return {"capture_candidate": raw if raw else None}


def validate_node(state: OnboardingState) -> dict[str, Any]:
    target = state.get("capture_target")
    candidate = state.get("capture_candidate")

    if target is None:
        if _first_missing(state) is None and state.get("stage") == "data_capture":
            return {
                "stage": "doc_verification",
                "progress": 25,
                "capture_candidate": None,
                "capture_error": None,
                "capture_target": None,
                "messages": list(state.get("messages", [])) + [
                    {
                        "role": "assistant",
                        "text": (
                            "Your details are saved. Next, we\u2019ll verify your "
                            "documents (PAN, Aadhaar, and selfie)."
                        ),
                    }
                ],
            }
        return {}

    if candidate is None:
        if not _last_user_text(state):
            return {}
        return {
            "capture_error": "empty",
            "capture_candidate": None,
            "messages": list(state.get("messages", [])) + [
                {
                    "role": "assistant",
                    "text": (
                        "I couldn\u2019t read a value for that. "
                        f"Please try again: {FIELD_QUESTIONS[target]}"
                    ),
                }
            ],
        }

    key = target
    value = candidate

    if key == "account_type":
        val = value.lower()
        if "saving" in val:
            normalized = "Savings"
        elif "current" in val or "checking" in val:
            normalized = "Current"
        else:
            return {
                "capture_error": "account_type",
                "capture_candidate": None,
                "messages": list(state.get("messages", [])) + [
                    {
                        "role": "assistant",
                        "text": "Please choose Savings or Current account.",
                    },
                ],
            }
        return _apply_success(state, key, normalized)

    if key == "full_name":
        ok, result = validate_name(value)
        if not ok:
            return _validation_fail(result, state)
        return _apply_success(state, key, result)

    if key == "dob":
        ok, result = validate_date(value)
        if not ok:
            return _validation_fail(result, state)
        return _apply_success(state, key, result)

    if key == "pan_number":
        ok, result = validate_pan(value)
        if not ok:
            return _validation_fail(result, state)
        return _apply_success(state, key, result)

    if key == "address":
        cleaned = value.strip()
        return _apply_success(state, key, cleaned)

    return {}


def _validation_fail(message: str, state: OnboardingState) -> dict[str, Any]:
    return {
        "capture_error": "validation",
        "capture_candidate": None,
        "messages": list(state.get("messages", [])) + [{"role": "assistant", "text": message}],
    }


def _apply_success(
    state: OnboardingState,
    field_key: str,
    normalized: str,
) -> dict[str, Any]:
    vals: dict[str, Optional[str]] = {k: state.get(k) for k in FIELD_ORDER}  # type: ignore[misc]
    vals[field_key] = normalized

    next_missing: Optional[str] = None
    for k in FIELD_ORDER:
        if not _is_set(vals.get(k)):
            next_missing = k
            break

    filled_count = sum(1 for k in FIELD_ORDER if _is_set(vals.get(k)))
    progress = 25 if next_missing is None else min(24, filled_count * 5)

    merged: dict[str, Any] = {
        field_key: normalized,
        "capture_candidate": None,
        "capture_error": None,
        "capture_target": None,
        "progress": progress,
    }

    if next_missing is None:
        merged["stage"] = "doc_verification"
        merged["progress"] = 25
        merged["messages"] = list(state.get("messages", [])) + [
            {
                "role": "assistant",
                "text": (
                    "Your details are saved. Next, we\u2019ll verify your "
                    "documents (PAN, Aadhaar, and selfie)."
                ),
            }
        ]
        return merged

    merged["messages"] = list(state.get("messages", [])) + [{"role": "assistant", "text": FIELD_QUESTIONS[next_missing]}]
    return merged


def build_data_capture_graph() -> CompiledStateGraph:
    workflow = StateGraph(OnboardingState)
    workflow.add_node("entry_node", entry_node)
    workflow.add_node("capture_node", capture_node)
    workflow.add_node("validate_node", validate_node)
    workflow.add_edge(START, "entry_node")
    workflow.add_edge("entry_node", "capture_node")
    workflow.add_edge("capture_node", "validate_node")
    workflow.add_edge("validate_node", END)
    return workflow.compile()


data_capture_graph = build_data_capture_graph()
