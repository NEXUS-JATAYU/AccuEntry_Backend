"""
Agent memory manager — stores and retrieves past interactions.

Current implementation: in-memory list (process-lifetime).
Swap for a vector DB (ChromaDB / Qdrant) in production for real
semantic similarity retrieval.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class AgentMemoryManager:

    def __init__(self) -> None:
        self._store: list[dict[str, Any]] = []

    def store_interaction(
        self,
        session_id: str,
        agent_name: str,
        input_data: dict[str, Any],
        output_data: dict[str, Any],
        risk_score: float | None = None,
        decision: str | None = None,
    ) -> None:
        self._store.append({
            "session_id": session_id,
            "agent_name": agent_name,
            "input_data": input_data,
            "output_data": output_data,
            "risk_score": risk_score,
            "decision": decision,
        })

    def retrieve_similar(
        self,
        agent_name: str,
        query_data: dict[str, Any],
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        matches = [
            entry for entry in self._store
            if entry["agent_name"] == agent_name
        ]
        return list(matches[-top_k:])
