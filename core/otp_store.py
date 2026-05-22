"""Redis-backed OTP store with in-memory fallback."""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any

OTP_EXPIRY_SECONDS = int(os.getenv("OTP_EXPIRY_SECONDS", "600"))
USE_REDIS_OTP = os.getenv("USE_REDIS_OTP", "true").lower() in {"1", "true", "yes"}

_redis = None
_memory: dict[str, dict[str, Any]] = {}


@dataclass
class OTPRecord:
    hashed_code: str
    created_at: float
    attempts: int = 0
    used: bool = False
    send_count: int = 1
    send_timestamps: list[float] = field(default_factory=list)


def _otp_key(session_id: str) -> str:
    return f"accuentry:otp:{session_id}"


def _get_redis():
    global _redis
    if _redis is not None or not USE_REDIS_OTP:
        return _redis
    url = os.getenv("REDIS_URL", "").strip()
    if not url:
        return None
    try:
        import redis

        _redis = redis.from_url(url, decode_responses=True)
    except Exception:
        _redis = None
    return _redis


def get_record(session_id: str) -> OTPRecord | None:
    r = _get_redis()
    if r is not None:
        try:
            raw = r.get(_otp_key(session_id))
            if raw:
                data = json.loads(raw)
                return OTPRecord(**data)
        except Exception:
            pass
    raw_mem = _memory.get(session_id)
    if raw_mem:
        return OTPRecord(**raw_mem)
    return None


def set_record(session_id: str, record: OTPRecord) -> None:
    payload = asdict(record)
    r = _get_redis()
    if r is not None:
        try:
            r.set(_otp_key(session_id), json.dumps(payload), ex=OTP_EXPIRY_SECONDS + 300)
            return
        except Exception:
            pass
    _memory[session_id] = payload


def delete_record(session_id: str) -> None:
    r = _get_redis()
    if r is not None:
        try:
            r.delete(_otp_key(session_id))
        except Exception:
            pass
    _memory.pop(session_id, None)
