from core.database import Base
from sqlalchemy import Boolean, Column, Integer, JSON, String, TIMESTAMP, Text, UniqueConstraint, text


class AccountTypeCapture(Base):
    __tablename__ = "account_type_capture"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(255), unique=True, index=True, nullable=False)
    account_type = Column(String(100), nullable=True)
    created_at = Column(TIMESTAMP(timezone=True), server_default=text("now()"))
    updated_at = Column(TIMESTAMP(timezone=True), server_default=text("now()"), onupdate=text("now()"))


class LLMDecisionLog(Base):
    __tablename__ = "llm_decision_logs"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(255), index=True, nullable=False)
    audit_session_id = Column(String(255), index=True, nullable=True)
    stage = Column(String(100), index=True, nullable=True)
    event_type = Column(String(120), index=True, nullable=False)
    decision_source = Column(String(40), nullable=True)
    decision = Column(String(80), nullable=True)
    friendly_text = Column(Text, nullable=False)
    log_hash = Column(String(64), index=True, nullable=False)
    input_payload_json = Column(JSON, nullable=True)
    output_payload_json = Column(JSON, nullable=True)
    metadata_json = Column(JSON, nullable=True)
    created_at = Column(TIMESTAMP(timezone=True), server_default=text("now()"))


class ComplianceStageAlert(Base):
    __tablename__ = "compliance_stage_alerts"
    __table_args__ = (
        UniqueConstraint("session_id", "stage", name="uq_compliance_stage_alert_session_stage"),
    )

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(255), index=True, nullable=False)
    audit_session_id = Column(String(255), index=True, nullable=True)
    stage = Column(String(100), index=True, nullable=False)
    alert_count = Column(Integer, nullable=False, server_default=text("0"))
    alert_summary = Column(Text, nullable=True)
    alert_details_json = Column(JSON, nullable=True)
    overall_flagged = Column(Boolean, nullable=False, server_default=text("false"))
    created_at = Column(TIMESTAMP(timezone=True), server_default=text("now()"))
    updated_at = Column(TIMESTAMP(timezone=True), server_default=text("now()"), onupdate=text("now()"))
