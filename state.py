"""
Shared onboarding state used by all agents and the supervisor graph.

Every agent subgraph accepts and returns this single TypedDict.
The supervisor routes between agents based on the `stage` field.
"""

import operator
from typing import Annotated, Optional, Literal
from typing_extensions import TypedDict


class OnboardingState(TypedDict):
    # ── Session ──────────────────────────────────────────────
    session_id: str

    # ── Conversation history ─────────────────────────────────
    # Each dict has: {"role": "user" | "assistant", "text": str}
    messages: Annotated[list[dict], operator.add]

    # ── Current stage (drives supervisor routing) ────────────
    stage: Literal[
        "data_capture",
        "doc_verification",
        "kyc_approval",
        "aml_screening",
        "fraud_check",
        "complete",
        "rejected",
    ]

    # ── Data capture fields ──────────────────────────────────
    full_name: Optional[str]
    dob: Optional[str]
    pan_number: Optional[str]
    address: Optional[str]
    account_type: Optional[str]

    # ── Document verification results ────────────────────────
    pan_verified: Optional[bool]
    aadhaar_verified: Optional[bool]
    face_verified: Optional[bool]

    # ── KYC approval ─────────────────────────────────────────
    kyc_status: Optional[str]       # "approved" | "rejected"

    # ── AML screening ────────────────────────────────────────
    aml_status: Optional[str]       # "pending" | "checking" | "clear" | "flagged"
    aml_raw_results: Optional[dict]  # raw output from all 4 tool calls
    aml_risk_score:  Optional[int]
    aml_in_background: bool
    aml_completed: bool

    # ── Fraud check ──────────────────────────────────────────
    fraud_status: Optional[str]     # "clear" | "flagged"

    # ── Progress (0–100, sent to frontend) ───────────────────
    progress: int

    # ── Signals React to show file-upload UI ─────────────────
    requires_upload: bool

    # ── Data capture subgraph scratch (optional between turns) ─
    capture_target: Optional[str]
    capture_candidate: Optional[str]
    capture_error: Optional[str]

    # ── Doc verification subgraph (optional routing / failure handling) ─
    doc_failure_type: Optional[str]  # "pan" | "aadhaar" | "face" when routing to failure_node
