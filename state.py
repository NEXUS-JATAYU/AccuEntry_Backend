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
        "decision_agent",
        "manual_review",
        "otp_verification",
        "complete",
        "rejected",
        "pending_docs",
        "escalated",
    ]

    # ── Data capture fields ──────────────────────────────────
    full_name: Optional[str]
    dob: Optional[str]
    gender: Optional[str]
    marital_status: Optional[str]
    pan_number: Optional[str]
    nationality: Optional[str]
    occupation_type: Optional[str]
    annual_income: Optional[str]
    source_of_funds: Optional[str]
    politically_exposed: Optional[str]
    mobile_number: Optional[str]
    email_id: Optional[str]
    id_proof_type: Optional[str]
    id_proof_number: Optional[str]
    address: Optional[str]
    account_type: Optional[str]
    mode_of_operation: Optional[str]
    debit_card_required: Optional[str]
    internet_banking: Optional[str]
    mobile_banking: Optional[str]
    sms_alerts: Optional[str]
    cheque_book: Optional[str]
    nominee_name: Optional[str]
    nominee_relationship: Optional[str]
    nominee_dob: Optional[str]

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
    fraud_risk_score: Optional[int]
    fraud_signals: list[str]
    fraud_reasoning: Optional[str]
    metadata: dict

    # ── Decision agent ───────────────────────────────────────
    decision_reason: Optional[str]
    decision_action: Optional[str]
    pending_docs: list[str]
    admin_override: bool
    audit_session_id: str

    # ── Upstream signals for decision agent ───────────────────
    video_kyc_status: Optional[str]       # "verified" | "failed" | "pending" | None
    risk_model_label: Optional[str]       # "clear" | "review" | "flagged"
    risk_model_confidence: Optional[float]  # 0.0–1.0
    kyc_data: Optional[dict]              # full_name, dob, pan, aadhaar, etc.

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
