"""
Audit logger for onboarding events.

Logs every agent decision and state transition to a structured JSON file.
Swap the file handler for a database writer in production.
"""

from __future__ import annotations

import json
import hashlib
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

_LOG_DIR = Path(os.getenv("AUDIT_LOG_DIR", "logs/audit"))

_STAGE_LABELS: dict[str, str] = {
    "data_capture": "Data Capture",
    "doc_verification": "Document Verification",
    "kyc_approval": "KYC Approval",
    "aml_screening": "AML Screening",
    "fraud_check": "Fraud Check",
    "manual_review": "Manual Review",
    "pending_docs": "Pending Documents",
    "escalated": "Compliance Escalation",
    "otp_verification": "OTP Verification",
    "complete": "Completed",
    "rejected": "Rejected",
    "decision_agent": "Decision Agent",
}

_UNDECIDED_PROGRESS: dict[str, str] = {
    "data_capture": "Collecting application details.",
    "doc_verification": "Verifying submitted documents.",
    "fraud_check": "Running fraud checks.",
    "kyc_approval": "Reviewing KYC information.",
    "aml_screening": "Running AML screening.",
    "manual_review": "Awaiting manual review.",
    "pending_docs": "Waiting for additional documents.",
    "otp_verification": "Awaiting OTP verification.",
}

_PREAMBLE_RE = re.compile(
    r"^(here are|conversion|converted|bullet points?|human-?readable|notes?:)\b",
    re.IGNORECASE,
)

_LLM_SYSTEM_PROMPT = (
    "You rewrite banking compliance decision logs for HITL reviewers reading a customer case timeline. "
    "Return exactly one factual sentence, max 120 characters. "
    "No bullet points, no preamble, no quotes, no invented facts."
)


def stage_display_label(stage: str | None) -> str:
    if not stage:
        return "Unknown"
    key = str(stage).strip().lower()
    if key in _STAGE_LABELS:
        return _STAGE_LABELS[key]
    return key.replace("_", " ").replace("-", " ").strip().title() or "Unknown"


def undecided_progress_line(stage: str | None) -> str:
    key = str(stage or "data_capture").strip().lower()
    if key in _UNDECIDED_PROGRESS:
        return _UNDECIDED_PROGRESS[key]
    return f"{stage_display_label(key)}: review in progress."


def normalize_audit_display_text(raw: str | None, *, max_len: int = 140) -> str:
    if not raw:
        return ""

    text = str(raw).strip()
    if not text:
        return ""

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    bullets: list[str] = []
    prose_parts: list[str] = []

    for line in lines:
        if _PREAMBLE_RE.match(line):
            continue
        if line.startswith("- "):
            bullets.append(line[2:].strip())
        elif line.startswith("-"):
            bullets.append(line[1:].strip())
        else:
            prose_parts.append(line)

    if bullets:
        first = bullets[0]
        if len(bullets) > 1 and len(first) < 80:
            second = bullets[1]
            if len(second) < 60:
                candidate = f"{first.rstrip('.')}; {second.rstrip('.')}."
                text = candidate if len(candidate) <= max_len else first
            else:
                text = first
        else:
            text = first
    elif prose_parts:
        text = prose_parts[0]
        for extra in prose_parts[1:]:
            if not _PREAMBLE_RE.match(extra) and not extra.startswith("- "):
                break
    else:
        text = re.sub(r"\s+", " ", text)

    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_len:
        text = text[: max_len - 1].rstrip() + "…"
    return text


def audit_fallback_line(
    *,
    stage: str | None,
    event_type: str | None,
    decision: str | None,
    output_payload: dict[str, Any] | None = None,
) -> str:
    output = output_payload or {}
    action = (output.get("action") or decision or "undecided").strip().lower()
    stage_key = str(stage or output.get("stage") or "unknown").strip().lower()

    if action == "undecided":
        return undecided_progress_line(stage_key)

    stage_name = stage_display_label(stage_key)
    reason = str(output.get("reason") or "").strip()
    if action == "approve":
        if reason and reason.lower() not in {"no reason was provided", "n/a"}:
            short = normalize_audit_display_text(reason, max_len=80)
            return short or f"Approved at {stage_name}."
        return f"Approved at {stage_name}."
    if action in {"reject", "rejected"}:
        return f"Rejected at {stage_name}."
    if action in {"queue_for_review", "manual_review"}:
        return f"Queued for manual review at {stage_name}."

    event_key = str(event_type or "").strip().lower()
    if event_key == "decision_agent_start":
        return "Decision review started."
    if event_key == "decision_agent_complete":
        return f"Decision recorded at {stage_name}."
    if event_key == "decision_agent_error":
        return "Decision error; routed to manual review."

    return f"{stage_name} — {action.replace('_', ' ').title()}."


def resolve_audit_display_line(
    *,
    friendly_text: str | None,
    stage: str | None,
    event_type: str | None,
    decision: str | None,
    output_payload: dict[str, Any] | None = None,
) -> str:
    normalized = normalize_audit_display_text(friendly_text)
    if normalized:
        return normalized
    return audit_fallback_line(
        stage=stage,
        event_type=event_type,
        decision=decision,
        output_payload=output_payload,
    )


def resolve_audit_status(
    *,
    decision: str | None,
    output_payload: dict[str, Any] | None = None,
    decision_source: str | None = None,
) -> str:
    output = output_payload or {}
    action = str(output.get("action") or decision or "undecided").strip().lower()
    if action and action != "undecided":
        return action
    source = str(decision_source or "").strip().lower()
    if source == "pending":
        return "pending"
    return "undecided"


def dedupe_adjacent_audit_logs(logs: list[dict[str, Any]], *, window_seconds: float = 10.0) -> list[dict[str, Any]]:
    if len(logs) < 2:
        return logs

    def _ts(entry: dict[str, Any]) -> float | None:
        raw = entry.get("created_at")
        if not raw:
            return None
        try:
            return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None

    skip_ids: set[Any] = set()
    for idx in range(1, len(logs)):
        prev = logs[idx - 1]
        curr = logs[idx]
        if curr.get("event_type") != "decision_summary":
            continue
        if prev.get("event_type") != "decision_agent_complete":
            continue

        prev_out = prev.get("output_payload") or {}
        curr_out = curr.get("output_payload") or {}
        prev_action = str(prev_out.get("action") or prev.get("decision") or "").lower()
        curr_action = str(curr_out.get("action") or curr.get("decision") or "").lower()
        if prev_action != curr_action or not prev_action:
            continue
        if str(prev.get("stage") or "") != str(curr.get("stage") or ""):
            continue

        t_prev, t_curr = _ts(prev), _ts(curr)
        if t_prev is not None and t_curr is not None and abs(t_curr - t_prev) <= window_seconds:
            skip_ids.add(curr.get("id"))

    return [entry for entry in logs if entry.get("id") not in skip_ids]


class AuditLogger:

    def __init__(self, log_dir: Path | None = None) -> None:
        self._log_dir = log_dir or _LOG_DIR
        self._log_dir.mkdir(parents=True, exist_ok=True)

    def log_event(
        self,
        session_id: str,
        event_type: str,
        *,
        input_data: dict[str, Any] | None = None,
        output_data: dict[str, Any] | None = None,
        decision: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "event_type": event_type,
            "input_data": input_data,
            "output_data": output_data,
            "decision": decision,
            "metadata": metadata,
        }
        try:
            safe_id = "".join(c for c in session_id if c.isalnum() or c == "-")
            log_file = self._log_dir / f"{safe_id}.jsonl"
            with open(log_file, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, default=str) + "\n")
        except OSError as exc:
            logger.warning("Audit log write failed for %s: %s", session_id, exc)

        self._write_db_log(
            session_id=session_id,
            event_type=event_type,
            input_data=input_data,
            output_data=output_data,
            decision=decision,
            metadata=metadata,
        )

    def _write_db_log(
        self,
        *,
        session_id: str,
        event_type: str,
        input_data: dict[str, Any] | None,
        output_data: dict[str, Any] | None,
        decision: str | None,
        metadata: dict[str, Any] | None,
    ) -> None:
        try:
            from core.database import SessionLocal
            from models.compliance_logs import LLMDecisionLog

            stage = None
            if isinstance(output_data, dict):
                stage = output_data.get("stage")
            if not stage and isinstance(metadata, dict):
                stage = metadata.get("outcome_stage") or metadata.get("workflow_stage")

            decision_source = (metadata or {}).get("decision_source") if isinstance(metadata, dict) else None
            audit_session_id = (metadata or {}).get("audit_session_id") if isinstance(metadata, dict) else None
            friendly_text = self._build_friendly_text(
                event_type=event_type,
                stage=stage,
                decision=decision,
                input_data=input_data,
                output_data=output_data,
                metadata=metadata,
            )
            hash_payload = f"{session_id}|{stage or 'unknown'}|{event_type}|{friendly_text.strip().lower()}"
            log_hash = hashlib.sha256(hash_payload.encode("utf-8")).hexdigest()

            db = SessionLocal()
            try:
                db.add(
                    LLMDecisionLog(
                        session_id=session_id,
                        audit_session_id=str(audit_session_id) if audit_session_id else None,
                        stage=str(stage) if stage else None,
                        event_type=event_type,
                        decision_source=str(decision_source) if decision_source else None,
                        decision=decision,
                        friendly_text=friendly_text,
                        log_hash=log_hash,
                        input_payload_json=input_data,
                        output_payload_json=output_data,
                        metadata_json=metadata,
                    )
                )
                db.commit()
            finally:
                db.close()
        except Exception as exc:
            logger.warning("Audit DB write failed for %s: %s", session_id, exc)

    def _build_friendly_text(
        self,
        *,
        event_type: str,
        stage: str | None,
        decision: str | None,
        input_data: dict[str, Any] | None,
        output_data: dict[str, Any] | None,
        metadata: dict[str, Any] | None,
    ) -> str:
        stage_text = stage or "unknown stage"
        action = (output_data or {}).get("action") or decision
        if str(action or "").strip().lower() == "undecided":
            return undecided_progress_line(str(stage_text))

        base_text = ""
        if event_type == "decision_agent_start":
            fraud_score = (input_data or {}).get("fraud_score")
            aml_status = (input_data or {}).get("aml_status")
            base_text = (
                f"Decision review started (fraud score {fraud_score}, AML {aml_status})."
            )
        elif event_type == "decision_agent_complete":
            action_val = (output_data or {}).get("action") or decision or "review"
            reason = (output_data or {}).get("reason") or "No reason was provided."
            if str(action_val).strip().lower() == "approve":
                base_text = f"Approved: {reason}"
            else:
                base_text = f"Decision {action_val} at {stage_display_label(str(stage_text))}: {reason}"
        elif event_type == "decision_agent_error":
            err = (metadata or {}).get("error") if isinstance(metadata, dict) else None
            base_text = (
                f"Decision error at {stage_display_label(str(stage_text))}; "
                f"routed to manual review ({err or 'unknown'})."
            )
        else:
            base = f"Audit event {event_type} at {stage_display_label(str(stage_text))}."
            base_text = f"{base} Decision={decision}." if decision else base

        llm_text = self._llm_humanize_text(
            base_text=base_text,
            event_type=event_type,
            stage=stage_text,
            decision=decision,
            input_data=input_data,
            output_data=output_data,
            metadata=metadata,
        )
        raw = llm_text or base_text
        return normalize_audit_display_text(raw) or raw

    def _llm_humanize_text(
        self,
        *,
        base_text: str,
        event_type: str,
        stage: str,
        decision: str | None,
        input_data: dict[str, Any] | None,
        output_data: dict[str, Any] | None,
        metadata: dict[str, Any] | None,
    ) -> str | None:
        try:
            from llm_config import AgentLLM

            llm = AgentLLM().get_llm("decision_agent")
            prompt_payload = {
                "event_type": event_type,
                "stage": stage,
                "decision": decision,
                "input_data": input_data or {},
                "output_data": output_data or {},
                "metadata": metadata or {},
                "base_summary": base_text,
            }
            response = llm.invoke(
                [
                    SystemMessage(content=_LLM_SYSTEM_PROMPT),
                    HumanMessage(
                        content=(
                            "Summarize this event in one sentence for a compliance reviewer. "
                            "Do not invent facts.\n"
                            f"Event JSON: {json.dumps(prompt_payload, default=str)}"
                        )
                    ),
                ]
            )
            content = str(getattr(response, "content", "") or "").strip()
            if not content:
                return None
            return normalize_audit_display_text(content) or content
        except Exception as exc:
            logger.warning("LLM-friendly audit text generation failed: %s", exc)
            return None
