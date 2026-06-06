"""
Data capture LangGraph subgraph: collects onboarding fields into OnboardingState.

Flow per invoke: entry_node -> capture_node -> validate_node -> END.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from memory_manager import AgentMemoryManager
from agents.data_capture.data_capture_validators import (
    parse_skip, 
    validate_address,
    validate_amount,
    validate_choice,
    validate_date,
    validate_email,
    validate_id_proof_number,
    validate_mobile_number,
    validate_name,
    validate_pan,
    validate_yes_no,
)
from state import OnboardingState

_memory = AgentMemoryManager()

FIELD_ORDER: tuple[str, ...] = (
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

FIELD_QUESTIONS: dict[str, str] = {
    "full_name": "Enter your full name (as per ID proof):",
    "dob": "Enter your date of birth (YYYY-MM-DD):",
    "gender": "Enter your gender (Male / Female / Third Gender):",
    "marital_status": "Enter marital status (Married / Unmarried / Others):",
    "pan_number": "Enter your PAN number:",
    "nationality": "Are you an Indian resident? (Yes / No):",
    "occupation_type": "Enter occupation type (Pvt. Sector / Govt / Business / Student / Retired / Other):",
    "annual_income": "Enter your approximate annual income (in Rs.):",
    "source_of_funds": "Enter source of funds (Salary / Business Income / Agriculture / Investment / Pension / Others):",
    "politically_exposed": "Are you a Politically Exposed Person? (Yes / No / Related to one):",
    "mobile_number": "Enter your 10-digit mobile number:",
    "email_id": "Enter your email address:",
    "id_proof_type": "Enter ID proof type (Passport / Voter ID / Driving Licence / Aadhaar / NREGA Job Card):",
    "id_proof_number": "Enter the document number of your ID proof:",
    "address": "Enter your full address (street, city, district, state, PIN, country):",
    "account_type": "Enter account type (Savings / Current / Fixed Deposit / Recurring Deposit):",
    "mode_of_operation": "Enter mode of operation (Self / Either or Survivor / Former or Survivor / Jointly Operated):",
    "debit_card_required": "Do you need an ATM-cum-Debit card? (Yes / No):",
    "internet_banking": "Do you require Internet Banking? (Yes / No):",
    "mobile_banking": "Do you require Mobile Banking? (Yes / No):",
    "sms_alerts": "Do you want SMS alerts on your registered mobile? (Yes / No):",
    "cheque_book": "Do you require a Cheque Book? (Yes / No):",
    "nominee_name": "Enter nominee's full name (or type 'skip' to skip nomination):",
    "nominee_relationship": "Enter nominee's relationship with you (Spouse / Parent / Child / Sibling / Other):",
    "nominee_dob": "Enter nominee's date of birth (YYYY-MM-DD):",
}

CHOICES: dict[str, list[str]] = {
    "gender": ["Male", "Female", "Third Gender"],
    "marital_status": ["Married", "Unmarried", "Others"],
    "occupation_type": ["Pvt. Sector", "Govt", "Business", "Student", "Retired", "Other"],
    "source_of_funds": ["Salary", "Business Income", "Agriculture", "Investment", "Pension", "Others"],
    "politically_exposed": ["Yes", "No", "Related to one"],
    "id_proof_type": ["Passport", "Voter ID", "Driving Licence", "Aadhaar", "NREGA Job Card"],
    "account_type": ["Savings", "Current", "Fixed Deposit", "Recurring Deposit"],
    "mode_of_operation": ["Self", "Either or Survivor", "Former or Survivor", "Jointly Operated"],
    "nominee_relationship": ["Spouse", "Parent", "Child", "Sibling", "Other"],
    "nationality": ["Yes", "No"],
    "debit_card_required": ["Yes", "No"],
    "internet_banking": ["Yes", "No"],
    "mobile_banking": ["Yes", "No"],
    "sms_alerts": ["Yes", "No"],
    "cheque_book": ["Yes", "No"],
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


def _text_has_choice(user_text: str, choice: str) -> bool:
    lowered = f" {(user_text or '').strip().lower()} "
    needle = re.escape(choice.strip().lower())
    return bool(re.search(rf"(?<!\w){needle}(?!\w)", lowered))


def _rule_extract_candidate(target: str, user_text: str) -> str | None:
    text = (user_text or "").strip()
    lowered = text.lower()
    if not text:
        return None

    if target in CHOICES:
        for choice in CHOICES[target]:
            if _text_has_choice(lowered, choice):
                return choice
        if target == "account_type":
            if any(token in lowered for token in ("saving", "savings")):
                return "Savings"
            if "current" in lowered:
                return "Current"
            if "fixed deposit" in lowered or "fd" in lowered:
                return "Fixed Deposit"
            if "recurring deposit" in lowered or "rd" in lowered:
                return "Recurring Deposit"
        return None

    if target == "pan_number":
        candidate = re.sub(r"\s+", "", text).upper()
        if re.fullmatch(r"[A-Z]{5}[0-9]{4}[A-Z]", candidate):
            return candidate

    if target == "mobile_number":
        digits = re.sub(r"\D", "", text)
        if len(digits) == 12 and digits.startswith("91"):
            digits = digits[2:]
        if len(digits) == 10:
            return digits

    if target == "email_id" and "@" in text:
        return text.lower().strip()

    if target == "dob" or target == "nominee_dob":
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text.strip()):
            return text.strip()

    return None


def entry_node(state: OnboardingState) -> dict[str, Any]:
    target = _first_missing(state)
    out: dict[str, Any] = {"capture_target": target, "capture_error": None}

    if target is None:
        return out

    current_target = state.get("capture_target")
    if current_target == target:
        return out

    msgs = state.get("messages", [])
    last_is_user = bool(msgs) and msgs[-1].get("role") == "user"
    if not last_is_user:
        out["messages"] = [{"role": "assistant", "text": FIELD_QUESTIONS[target]}]
    return out


# Yes/No fields: never send through the LLM — a paraphrase like "Yes, I am a resident"
# fails validate_yes_no() which only accepts plain yes/no tokens.
_SKIP_LLM_FOR_TARGETS: frozenset[str] = frozenset(
    {
        "nationality",
        "debit_card_required",
        "internet_banking",
        "mobile_banking",
        "sms_alerts",
        "cheque_book",
    }
)


async def capture_node(state: OnboardingState) -> dict[str, Any]:
    target = state.get("capture_target")
    if not target:
        return {"capture_candidate": None}

    user_text = _last_user_text(state)
    if not user_text:
        return {"capture_candidate": None}

    ruled_candidate = _rule_extract_candidate(target, user_text)
    if ruled_candidate is not None:
        return {"capture_candidate": ruled_candidate}

    # Keep the flow deterministic: free-text fields use the user's message as-is,
    # and validators decide whether it is acceptable.
    if target in _SKIP_LLM_FOR_TARGETS or target not in CHOICES:
        return {"capture_candidate": user_text}

    return {"capture_candidate": None}


def validate_node(state: OnboardingState) -> dict[str, Any]:
    target = state.get("capture_target")
    candidate = state.get("capture_candidate")

    if target is None:
        if _first_missing(state) is None and state.get("stage") == "data_capture":
            metadata = dict(state.get("metadata") or {})
            already_prompted = bool(metadata.get("details_confirmation_prompted"))
            metadata["details_confirmation_prompted"] = True

            if already_prompted:
                completion_message = "Your details are saved. Next, we will verify your documents (PAN, Aadhaar, and selfie)."
            else:
                completion_message = json.dumps(
                    {
                        "type": "DETAILS_CONFIRMATION_REQUIRED",
                        "channel": "chatbot",
                        "payload": {
                            "message": "please confirm your details before proceeding for identity verification",
                            "buttonLabel": "Edit Details",
                        },
                    }
                )

            return {
                "stage": "doc_verification",
                "progress": 25,
                "capture_candidate": None,
                "capture_error": None,
                "capture_target": None,
                "metadata": metadata,
                "messages": [
                    {
                        "role": "assistant",
                        "text": completion_message,
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
            "messages": [
                {
                    "role": "assistant",
                    "text": f"I could not read a value for that. Please try again: {FIELD_QUESTIONS[target]}",
                }
            ],
        }

    key = target
    value = candidate

    if key == "full_name":
        ok, result = validate_name(value)
        if not ok:
            return _validation_fail(result, state)
        return _apply_success(state, {key: result})

    if key == "dob":
        ok, result = validate_date(value)
        if not ok:
            return _validation_fail(result, state)
        return _apply_success(state, {key: result})

    if key in ("gender", "marital_status", "occupation_type", "source_of_funds", "id_proof_type", "account_type", "mode_of_operation", "nominee_relationship"):
        ok, result = validate_choice(value, CHOICES[key], key.replace("_", " "))
        if not ok:
            return _validation_fail(result, state)
        return _apply_success(state, {key: result})

    if key == "pan_number":
        ok, result = validate_pan(value)
        if not ok:
            return _validation_fail(result, state)
        return _apply_success(state, {key: result})

    if key == "nationality":
        ok, result = validate_yes_no(value, "nationality")
        if not ok:
            return _validation_fail(result, state)
        return _apply_success(state, {key: result})

    if key == "annual_income":
        ok, result = validate_amount(value, "annual income")
        if not ok:
            return _validation_fail(result, state)
        return _apply_success(state, {key: result})

    if key == "politically_exposed":
        ok, result = validate_choice(value, CHOICES[key], "politically exposed")
        if not ok:
            return _validation_fail(result, state)
        return _apply_success(state, {key: result})

    if key == "mobile_number":
        ok, result = validate_mobile_number(value)
        if not ok:
            return _validation_fail(result, state)
        return _apply_success(state, {key: result})

    if key == "email_id":
        ok, result = validate_email(value)
        if not ok:
            return _validation_fail(result, state)
        return _apply_success(state, {key: result})

    if key == "id_proof_number":
        ok, result = validate_id_proof_number(value)
        if not ok:
            return _validation_fail(result, state)
        return _apply_success(state, {key: result})

    if key == "address":
        ok, result = validate_address(value)
        if not ok:
            return _validation_fail(result, state)
        return _apply_success(state, {key: result})

    if key in ("debit_card_required", "internet_banking", "mobile_banking", "sms_alerts", "cheque_book"):
        ok, result = validate_yes_no(value, key.replace("_", " "))
        if not ok:
            return _validation_fail(result, state)
        return _apply_success(state, {key: result})

    if key == "nominee_name":
        if parse_skip(value) == "skip":
            return _apply_success(
                state,
                {
                    "nominee_name": "SKIPPED",
                    "nominee_relationship": "SKIPPED",
                    "nominee_dob": "SKIPPED",
                },
            )
        ok, result = validate_name(value)
        if not ok:
            return _validation_fail(result, state)
        return _apply_success(state, {key: result})

    if key == "nominee_dob":
        ok, result = validate_date(value)
        if not ok:
            return _validation_fail(result, state)
        return _apply_success(state, {key: result})

    return {}


def _validation_fail(message: str, state: OnboardingState) -> dict[str, Any]:
    session_id = state.get("audit_session_id") or state.get("session_id") or "unknown"
    target = state.get("capture_target") or "unknown"
    attempted_value = state.get("capture_candidate") or _last_user_text(state)
    _memory.store_interaction(
        session_id=session_id,
        agent_name="data_capture",
        input_data={
            "field": target,
            "attempted_value": attempted_value,
            "user_text": _last_user_text(state),
        },
        output_data={
            "status": "validation_failed",
            "message": message,
        },
        decision="validation_failed",
        metadata={
            "audit_session_id": state.get("audit_session_id") or session_id,
            "workflow_stage": state.get("stage") or "data_capture",
            "field": target,
        },
        event_type="data_capture_validation_fail",
        doc_suffix=str(target),
    )
    return {
        "capture_error": "validation",
        "capture_candidate": None,
        "messages": [{"role": "assistant", "text": message}],
    }


def _apply_success(state: OnboardingState, updates: dict[str, str]) -> dict[str, Any]:
    vals: dict[str, Optional[str]] = {k: state.get(k) for k in FIELD_ORDER}  # type: ignore[misc]
    for k, v in updates.items():
        vals[k] = v

    next_missing: Optional[str] = None
    for k in FIELD_ORDER:
        if not _is_set(vals.get(k)):
            next_missing = k
            break

    filled_count = sum(1 for k in FIELD_ORDER if _is_set(vals.get(k)))
    progress = 25 if next_missing is None else int((filled_count / len(FIELD_ORDER)) * 24)

    merged: dict[str, Any] = {
        **updates,
        "capture_candidate": None,
        "capture_error": None,
        "capture_target": next_missing,
        "progress": progress,
    }

    session_id = state.get("audit_session_id") or state.get("session_id") or "unknown"
    _memory.store_interaction(
        session_id=session_id,
        agent_name="data_capture",
        input_data={
            "field": state.get("capture_target"),
            "attempted_value": state.get("capture_candidate") or _last_user_text(state),
            "updates": updates,
        },
        output_data={
            "status": "accepted",
            "next_missing": next_missing,
            "next_stage": "doc_verification" if next_missing is None else "data_capture",
            "progress": 25 if next_missing is None else progress,
        },
        decision="accepted",
        metadata={
            "audit_session_id": state.get("audit_session_id") or session_id,
            "workflow_stage": state.get("stage") or "data_capture",
            "field_count_updated": len(updates),
        },
        event_type="data_capture_field_update",
        doc_suffix=str(state.get("capture_target") or "field"),
    )

    if next_missing is None:
        metadata = dict(state.get("metadata") or {})
        already_prompted = bool(metadata.get("details_confirmation_prompted"))
        metadata["details_confirmation_prompted"] = True

        if already_prompted:
            completion_message = "Your details are saved. Next, we will verify your documents (PAN, Aadhaar, and selfie)."
        else:
            completion_message = json.dumps(
                {
                    "type": "DETAILS_CONFIRMATION_REQUIRED",
                    "channel": "chatbot",
                    "payload": {
                        "message": "PLEASE CONFIRM YOUR DETAILS",
                        "buttonLabel": "VIEW DETAILS",
                    },
                }
            )

        merged["stage"] = "doc_verification"
        merged["progress"] = 25
        merged["metadata"] = metadata
        merged["messages"] = [
            {
                "role": "assistant",
                "text": completion_message,
            }
        ]
        return merged

    merged["messages"] = [
        {"role": "assistant", "text": FIELD_QUESTIONS[next_missing]}
    ]

    if next_missing == "mode_of_operation":
        metadata = dict(state.get("metadata") or {})
        if not metadata.get("details_confirmation_prompted"):
            metadata["details_confirmation_prompted"] = True
            merged["metadata"] = metadata
            merged["messages"] = [
                {"role": "assistant", "text": FIELD_QUESTIONS[next_missing]},
                {
                    "role": "assistant",
                    "text": json.dumps(
                        {
                            "type": "DETAILS_CONFIRMATION_REQUIRED",
                            "channel": "chatbot",
                            "payload": {
                                "message": "PLEASE CONFIRM YOUR DETAILS",
                                "buttonLabel": "VIEW DETAILS",
                            },
                        }
                    ),
                },
            ]

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
