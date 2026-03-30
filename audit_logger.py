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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

_LOG_DIR = Path(os.getenv("AUDIT_LOG_DIR", "logs/audit"))


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
        base_text = ""
        if event_type == "decision_agent_start":
            fraud_score = (input_data or {}).get("fraud_score")
            aml_status = (input_data or {}).get("aml_status")
            base_text = (
                f"Decision engine started at {stage_text}. "
                f"Fraud score={fraud_score}, AML status={aml_status}."
            )
        elif event_type == "decision_agent_complete":
            action = (output_data or {}).get("action") or decision or "review"
            reason = (output_data or {}).get("reason") or "No reason was provided."
            source = (metadata or {}).get("decision_source") if isinstance(metadata, dict) else None
            source_text = f" Source={source}." if source else ""
            base_text = (
                f"Decision engine completed at {stage_text}. "
                f"Action={action}. Reason={reason}.{source_text}"
            )
        elif event_type == "decision_agent_error":
            err = (metadata or {}).get("error") if isinstance(metadata, dict) else None
            base_text = (
                f"Decision engine encountered an error at {stage_text}. "
                f"System routed this case for manual review. Details={err or 'n/a'}."
            )
        else:
            base = f"Audit event {event_type} recorded at {stage_text}."
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
        return llm_text or base_text

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
                    SystemMessage(
                        content=(
                            "You rewrite banking compliance decision logs into short, human-friendly text. "
                            "Return exactly 2-4 bullet points using '-' prefix. "
                            "Keep it factual, concise, and avoid jargon."
                        )
                    ),
                    HumanMessage(
                        content=(
                            "Convert this event into readable bullet points for compliance dashboard users. "
                            "Do not invent facts.\n"
                            f"Event JSON: {json.dumps(prompt_payload, default=str)}"
                        )
                    ),
                ]
            )
            content = str(getattr(response, "content", "") or "").strip()
            if not content:
                return None
            return content
        except Exception as exc:
            logger.warning("LLM-friendly audit text generation failed: %s", exc)
            return None
