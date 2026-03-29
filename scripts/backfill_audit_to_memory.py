"""
Backfill audit JSONL events into agent memory collections.

Usage:
  python -m scripts.backfill_audit_to_memory
  python -m scripts.backfill_audit_to_memory --log-dir logs/audit --limit 5000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from memory_manager import AgentMemoryManager


def _agent_from_event(event_type: str) -> str:
    e = (event_type or "").lower()
    if "decision" in e or "review" in e or "rejection" in e or "escalat" in e:
        return "decision_agent"
    if "aml" in e:
        return "aml_screening"
    if "fraud" in e:
        return "fraud_check"
    if "doc" in e or "kyc" in e or "verify" in e:
        return "doc_verify"
    if "capture" in e:
        return "data_capture"
    return "decision_agent"


def _safe_read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    parsed = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    out.append(parsed)
    except OSError:
        return []
    return out


def backfill(log_dir: Path, limit: int = 0) -> tuple[int, int]:
    memory = AgentMemoryManager()
    files = sorted(log_dir.glob("*.jsonl"))

    imported = 0
    skipped = 0
    for file_path in files:
        entries = _safe_read_jsonl(file_path)
        for entry in entries:
            if limit > 0 and imported >= limit:
                return imported, skipped

            session_id = str(entry.get("session_id") or "").strip()
            if not session_id:
                skipped += 1
                continue

            event_type = str(entry.get("event_type") or "audit_event")
            input_data = entry.get("input_data") if isinstance(entry.get("input_data"), dict) else {}
            output_data = entry.get("output_data") if isinstance(entry.get("output_data"), dict) else {}
            decision = entry.get("decision")
            metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
            metadata = {
                **metadata,
                "backfilled": True,
                "timestamp": entry.get("timestamp"),
            }

            memory.store_interaction(
                session_id=session_id,
                agent_name=_agent_from_event(event_type),
                input_data=input_data,
                output_data=output_data,
                decision=str(decision) if decision is not None else None,
                metadata=metadata,
                event_type=event_type,
                doc_suffix=str(entry.get("timestamp") or ""),
            )
            imported += 1

    return imported, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill audit logs into Chroma-backed memory.")
    parser.add_argument("--log-dir", default="logs/audit", help="Directory containing *.jsonl audit files")
    parser.add_argument("--limit", type=int, default=0, help="Max events to import (0 = no limit)")
    args = parser.parse_args()

    imported, skipped = backfill(Path(args.log_dir), limit=args.limit)
    print(f"Backfill complete. Imported={imported}, Skipped={skipped}")


if __name__ == "__main__":
    main()
