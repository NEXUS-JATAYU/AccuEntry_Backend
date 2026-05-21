from typing import Any

from pydantic import BaseModel, Field, field_validator

from core.security import sanitize_session_id, sanitize_user_input


class ChatRequest(BaseModel):
    user_input: str = ""
    session_id: str | None = None

    @field_validator("user_input")
    @classmethod
    def validate_user_input(cls, v: str) -> str:
        return sanitize_user_input(v, field_name="user_input")

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, v: str | None) -> str | None:
        if v is None or not str(v).strip():
            return None
        return sanitize_session_id(v) or None


class ChatResponse(BaseModel):
    message: str
    progress: int = 0
    requires_upload: bool = False
    stage: str = "data_capture"
    action: str | None = None
    completed: bool = False
    session_ended: bool = False
    session_end_reason: str | None = None
    step: int = 1
    current_main_step: str = "Detail Capture"
    aml_status: str | None = "pending"
    aml_in_background: bool = False
    fraud_status: str | None = None
    fraud_risk_score: int | None = None
    fraud_signals: list[str] = []
    fraud_reasoning: str | None = None
    otp_required: bool = False


class SessionDetailsResponse(BaseModel):
    session_id: str
    stage: str
    details: dict[str, str | None]


class SessionDetailsUpdateRequest(BaseModel):
    session_id: str
    details: dict[str, Any] = Field(default_factory=dict)

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, v: str) -> str:
        return sanitize_session_id(v)


class SessionDetailsUpdateResponse(BaseModel):
    session_id: str
    stage: str
    details: dict[str, str | None]
    message: str
    errors: dict[str, str] = Field(default_factory=dict)
