"""
Audit logger for onboarding events.

Logs every agent decision and state transition to a structured JSON file.
Swap the file handler for a database writer in production.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
