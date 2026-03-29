"""
Chroma memory backend for agent interactions.

This module is optional by design:
- If chromadb or embedding dependencies are unavailable, methods no-op.
- The rest of the app continues to run with in-memory fallback.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ChromaMemoryBackend:
    PII_FIELDS = {
        "pan_number",
        "id_proof_number",
        "aadhaar_number",
        "mobile_number",
        "phone",
        "phone_number",
        "email",
        "email_id",
    }

    def __init__(self) -> None:
        self.enabled = os.getenv("AGENT_MEMORY_PROVIDER", "chroma").lower() == "chroma"
        self.write_enabled = os.getenv("AGENT_MEMORY_WRITE_ENABLED", "true").lower() in {"1", "true", "yes"}
        self.retrieval_enabled = os.getenv("AGENT_MEMORY_RETRIEVAL_ENABLED", "true").lower() in {"1", "true", "yes"}
        self.reward_rerank_enabled = os.getenv("AGENT_MEMORY_REWARD_RERANK_ENABLED", "true").lower() in {"1", "true", "yes"}
        self.reward_alpha = float(os.getenv("AGENT_MEMORY_REWARD_ALPHA", "0.8"))
        self.reward_beta = float(os.getenv("AGENT_MEMORY_REWARD_BETA", "0.2"))
        self._client = None
        self._embedding_fn = None
        self._collections: dict[str, Any] = {}
        self._prefix = os.getenv("CHROMA_COLLECTION_PREFIX", "accuentry")
        if not self.enabled:
            return
        self._init_client()

    def _init_client(self) -> None:
        try:
            import chromadb
            from chromadb.utils import embedding_functions
        except Exception as exc:
            self.enabled = False
            logger.warning("Chroma backend disabled: dependency unavailable (%s)", exc)
            return

        host = os.getenv("CHROMA_HOST", "").strip()
        port = int(os.getenv("CHROMA_PORT", "8000"))
        persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./.chroma")

        try:
            if host:
                self._client = chromadb.HttpClient(host=host, port=port)
            else:
                self._client = chromadb.PersistentClient(path=persist_dir)
        except Exception as exc:
            self.enabled = False
            logger.warning("Chroma backend disabled: client init failed (%s)", exc)
            return

        model_name = os.getenv(
            "CHROMA_EMBED_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        )
        try:
            self._embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=model_name
            )
        except Exception as exc:
            # Keep backend active: Chroma can still use default embedding function.
            self._embedding_fn = None
            logger.warning("SentenceTransformer embedding init failed (%s); using default", exc)

    def _collection_name(self, agent_name: str) -> str:
        safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in agent_name)
        return f"{self._prefix}_{safe}".strip("_")

    def _get_collection(self, agent_name: str):
        if not self.enabled or self._client is None:
            return None
        name = self._collection_name(agent_name)
        existing = self._collections.get(name)
        if existing is not None:
            return existing
        kwargs: dict[str, Any] = {"name": name}
        if self._embedding_fn is not None:
            kwargs["embedding_function"] = self._embedding_fn
        collection = self._client.get_or_create_collection(**kwargs)
        self._collections[name] = collection
        return collection

    def _hash_value(self, value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]

    def _sanitize(self, payload: Any, parent_key: str | None = None) -> Any:
        if isinstance(payload, dict):
            clean: dict[str, Any] = {}
            for key, value in payload.items():
                if key in self.PII_FIELDS and isinstance(value, str) and value:
                    clean[key] = f"hash:{self._hash_value(value)}"
                else:
                    clean[key] = self._sanitize(value, parent_key=key)
            return clean
        if isinstance(payload, list):
            return [self._sanitize(item, parent_key=parent_key) for item in payload]
        if isinstance(payload, str) and parent_key in self.PII_FIELDS and payload:
            return f"hash:{self._hash_value(payload)}"
        return payload

    def _to_text(self, input_data: dict[str, Any], output_data: dict[str, Any]) -> str:
        safe_in = self._sanitize(input_data)
        safe_out = self._sanitize(output_data)
        return (
            "Input summary:\n"
            f"{json.dumps(safe_in, ensure_ascii=True, sort_keys=True)}\n"
            "Output summary:\n"
            f"{json.dumps(safe_out, ensure_ascii=True, sort_keys=True)}"
        )

    def _flat_metadata(self, payload: dict[str, Any]) -> dict[str, str | int | float | bool]:
        out: dict[str, str | int | float | bool] = {}
        for key, value in payload.items():
            if isinstance(value, (str, int, float, bool)):
                out[key] = value
            elif value is None:
                continue
            else:
                out[key] = json.dumps(self._sanitize(value), ensure_ascii=True, sort_keys=True)
        return out

    def _doc_id(self, session_id: str, agent_name: str, event_type: str, suffix: str = "") -> str:
        base = f"{session_id}:{agent_name}:{event_type}".replace(" ", "_")
        if suffix:
            return f"{base}:{suffix}"[:240]
        return base[:240]

    def _deterministic_suffix(
        self,
        *,
        input_data: dict[str, Any],
        output_data: dict[str, Any],
        metadata: dict[str, Any] | None,
    ) -> str:
        key_payload = {
            "input": self._sanitize(input_data),
            "output": self._sanitize(output_data),
            "metadata": self._sanitize(metadata or {}),
        }
        stable = json.dumps(key_payload, ensure_ascii=True, sort_keys=True)
        return hashlib.sha256(stable.encode("utf-8")).hexdigest()[:24]

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _feedback_by_session(self, collection, max_rows: int = 5000) -> dict[str, float]:
        try:
            payload = collection.get(
                where={"event_type": "feedback"},
                include=["metadatas"],
                limit=max_rows,
            )
        except Exception:
            return {}

        metadatas = payload.get("metadatas") or []
        rewards: dict[str, float] = {}
        for md in metadatas:
            if not isinstance(md, dict):
                continue
            sid = md.get("session_id")
            if not isinstance(sid, str) or not sid:
                continue
            score = self._safe_float(md.get("risk_score"), 0.0)
            prev = rewards.get(sid)
            if prev is None or score > prev:
                rewards[sid] = score
        return rewards

    def upsert_interaction(
        self,
        *,
        agent_name: str,
        session_id: str,
        input_data: dict[str, Any],
        output_data: dict[str, Any],
        risk_score: float | None = None,
        decision: str | None = None,
        metadata: dict[str, Any] | None = None,
        event_type: str = "interaction",
        doc_suffix: str = "",
    ) -> bool:
        if not self.write_enabled:
            return False
        collection = self._get_collection(agent_name)
        if collection is None:
            return False
        try:
            suffix = doc_suffix or self._deterministic_suffix(
                input_data=input_data,
                output_data=output_data,
                metadata=metadata,
            )
            doc_id = self._doc_id(session_id, agent_name, event_type, suffix=suffix)
            doc_text = self._to_text(input_data=input_data, output_data=output_data)
            raw_meta = {
                "agent_name": agent_name,
                "session_id": session_id,
                "event_type": event_type,
                "created_at": _iso_now(),
                "risk_score": float(risk_score) if risk_score is not None else 0.0,
                "decision": decision or "",
                "input_data_json": json.dumps(self._sanitize(input_data), ensure_ascii=True, sort_keys=True),
                "output_data_json": json.dumps(self._sanitize(output_data), ensure_ascii=True, sort_keys=True),
            }
            if metadata:
                raw_meta.update(self._sanitize(metadata))
            collection.upsert(
                ids=[doc_id],
                documents=[doc_text],
                metadatas=[self._flat_metadata(raw_meta)],
            )
            return True
        except Exception as exc:
            logger.warning("Chroma upsert failed for %s: %s", agent_name, exc)
            return False

    def query_similar(
        self,
        *,
        agent_name: str,
        query_data: dict[str, Any],
        top_k: int = 3,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if not self.retrieval_enabled:
            return []
        collection = self._get_collection(agent_name)
        if collection is None:
            return []
        try:
            query_text = json.dumps(self._sanitize(query_data), ensure_ascii=True, sort_keys=True)
            query_kwargs: dict[str, Any] = {
                "query_texts": [query_text],
                "n_results": max(5, top_k * 4),
            }
            if where:
                query_kwargs["where"] = where
            res = collection.query(**query_kwargs)

            feedback_rewards = self._feedback_by_session(collection) if self.reward_rerank_enabled else {}

            out: list[dict[str, Any]] = []
            ids = (res.get("ids") or [[]])[0]
            docs = (res.get("documents") or [[]])[0]
            metas = (res.get("metadatas") or [[]])[0]
            dists = (res.get("distances") or [[]])[0]
            for idx, doc_id in enumerate(ids):
                md = metas[idx] if idx < len(metas) else {}
                if isinstance(md, dict) and md.get("event_type") == "feedback":
                    continue
                output_data_json = md.get("output_data_json") if isinstance(md, dict) else None
                output_data = {}
                if isinstance(output_data_json, str) and output_data_json:
                    try:
                        output_data = json.loads(output_data_json)
                    except Exception:
                        output_data = {}

                distance = dists[idx] if idx < len(dists) else None
                similarity = 0.0
                if isinstance(distance, (int, float)):
                    similarity = 1.0 / (1.0 + float(distance))
                session_key = md.get("session_id") if isinstance(md, dict) else None
                reward = feedback_rewards.get(session_key, 0.0) if isinstance(session_key, str) else 0.0
                rerank_score = (self.reward_alpha * similarity) + (self.reward_beta * reward)
                out.append(
                    {
                        "id": doc_id,
                        "document": docs[idx] if idx < len(docs) else "",
                        "metadata": md,
                        "risk_score": md.get("risk_score") if isinstance(md, dict) else None,
                        "decision": md.get("decision") if isinstance(md, dict) else None,
                        "distance": distance,
                        "similarity_score": round(similarity, 6),
                        "reward_score": round(reward, 6),
                        "rerank_score": round(rerank_score, 6),
                        "output_data": output_data,
                    }
                )
            out.sort(key=lambda row: row.get("rerank_score", 0.0), reverse=True)
            return out[: max(1, top_k)]
        except Exception as exc:
            logger.warning("Chroma query failed for %s: %s", agent_name, exc)
            return []

    def upsert_feedback(
        self,
        *,
        agent_name: str,
        session_id: str,
        outcome: str,
        otp_no_rework: bool,
        reward_score: float,
        source: str,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        feedback_input = {
            "outcome": outcome,
            "otp_no_rework": otp_no_rework,
            "source": source,
        }
        feedback_output = {
            "reward_score": reward_score,
        }
        return self.upsert_interaction(
            agent_name=agent_name,
            session_id=session_id,
            input_data=feedback_input,
            output_data=feedback_output,
            risk_score=reward_score,
            decision=outcome,
            metadata=metadata or {},
            event_type="feedback",
            doc_suffix=source,
        )