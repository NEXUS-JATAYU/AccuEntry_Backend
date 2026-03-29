"""
Agent memory manager — stores and retrieves past interactions.

Current implementation: in-memory list (process-lifetime).
Swap for a vector DB (ChromaDB / Qdrant) in production for real
semantic similarity retrieval.
"""

from __future__ import annotations

import logging
from typing import Any

from core.chroma_memory import ChromaMemoryBackend

logger = logging.getLogger(__name__)


class AgentMemoryManager:

    def __init__(self) -> None:
        self._store: list[dict[str, Any]] = []
        self._chroma = ChromaMemoryBackend()

    def store_interaction(
        self,
        session_id: str,
        agent_name: str,
        input_data: dict[str, Any],
        output_data: dict[str, Any],
        risk_score: float | None = None,
        decision: str | None = None,
        metadata: dict[str, Any] | None = None,
        event_type: str = "interaction",
        doc_suffix: str = "",
    ) -> None:
        record = {
            "session_id": session_id,
            "agent_name": agent_name,
            "input_data": input_data,
            "output_data": output_data,
            "risk_score": risk_score,
            "decision": decision,
            "metadata": metadata or {},
            "event_type": event_type,
        }
        self._store.append(record)

        ok = self._chroma.upsert_interaction(
            agent_name=agent_name,
            session_id=session_id,
            input_data=input_data,
            output_data=output_data,
            risk_score=risk_score,
            decision=decision,
            metadata=metadata,
            event_type=event_type,
            doc_suffix=doc_suffix,
        )
        if not ok:
            logger.debug("Chroma upsert skipped/failed for agent=%s session=%s", agent_name, session_id)

    def retrieve_similar(
        self,
        agent_name: str,
        query_data: dict[str, Any],
        top_k: int = 3,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        chroma_matches = self._chroma.query_similar(
            agent_name=agent_name,
            query_data=query_data,
            top_k=top_k,
            where=where,
        )
        if chroma_matches:
            return chroma_matches

        matches = [
            entry for entry in self._store
            if entry["agent_name"] == agent_name
        ]
        return list(matches[-top_k:])

    def store_feedback(
        self,
        *,
        agent_name: str,
        session_id: str,
        outcome: str,
        otp_no_rework: bool,
        reward_score: float,
        source: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        feedback_record = {
            "session_id": session_id,
            "agent_name": agent_name,
            "input_data": {
                "outcome": outcome,
                "otp_no_rework": otp_no_rework,
                "source": source,
            },
            "output_data": {
                "reward_score": reward_score,
            },
            "risk_score": reward_score,
            "decision": outcome,
            "metadata": metadata or {},
            "event_type": "feedback",
        }
        self._store.append(feedback_record)
        self._chroma.upsert_feedback(
            agent_name=agent_name,
            session_id=session_id,
            outcome=outcome,
            otp_no_rework=otp_no_rework,
            reward_score=reward_score,
            source=source,
            metadata=metadata,
        )
