"""
Input sanitization, session validation, and optional API-key auth for AccuEntry APIs.
"""

from __future__ import annotations

import os
import re
import unicodedata
from typing import Annotated

from fastapi import Header, HTTPException, status

# ── Limits ────────────────────────────────────────────────────────────────────
MAX_USER_INPUT_LENGTH = int(os.getenv("MAX_USER_INPUT_LENGTH", "4000"))
MAX_SESSION_ID_LENGTH = 64
MIN_SESSION_ID_LENGTH = 8

# Control chars + bidi overrides often used in prompt injection
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_BIDI_OVERRIDE_RE = re.compile(r"[\u202a-\u202e\u2066-\u2069]")
_SESSION_ID_RE = re.compile(r"^[a-zA-Z0-9_-]+$")

# High-risk instruction patterns (defense in depth — not a full jailbreak filter)
_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
        r"disregard\s+(all\s+)?(previous|prior|system)\s+",
        r"you\s+are\s+now\s+(a|an)\s+",
        r"new\s+system\s+prompt",
        r"<\s*/?\s*system\s*>",
        r"```\s*system",
        r"jailbreak",
        r"reveal\s+(the\s+)?(system|hidden)\s+prompt",
        r"override\s+(safety|security|policy)",
    )
)

API_KEY = os.getenv("API_KEY", "").strip()
REQUIRE_API_KEY = os.getenv("REQUIRE_API_KEY", "false").lower() in {"1", "true", "yes"}
AGENT_SERVICE_API_KEY = os.getenv("AGENT_SERVICE_API_KEY", os.getenv("API_KEY", "")).strip()


def _normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def sanitize_user_input(raw: str | None, *, field_name: str = "user_input") -> str:
    """
    Strip, normalize, length-limit, and block obvious prompt-injection phrases.
    Raises HTTPException 400 on rejection.
    """
    if raw is None:
        return ""

    text = _normalize_unicode(str(raw)).strip()
    text = _CONTROL_CHAR_RE.sub("", text)
    text = _BIDI_OVERRIDE_RE.sub("", text)
    text = re.sub(r"\s{3,}", "  ", text)

    if len(text) > MAX_USER_INPUT_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{field_name} exceeds maximum length of {MAX_USER_INPUT_LENGTH} characters.",
        )

    lowered = text.lower()
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(lowered):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Message contains disallowed content. Please rephrase your request.",
            )

    return text


def sanitize_session_id(raw: str | None) -> str:
    """Validate client-supplied session id format."""
    if not raw:
        return ""
    sid = _normalize_unicode(str(raw)).strip()
    if len(sid) < MIN_SESSION_ID_LENGTH or len(sid) > MAX_SESSION_ID_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid session_id length.",
        )
    if not _SESSION_ID_RE.match(sid):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="session_id contains invalid characters.",
        )
    return sid


def wrap_user_message_for_llm(sanitized_text: str) -> str:
    """
    Delimit user content so downstream prompts can treat it as untrusted data.
    Agents should prefer structured extraction over echoing this block verbatim.
    """
    if not sanitized_text:
        return ""
    escaped = sanitized_text.replace("</user>", "< /user>")
    return f"<user>\n{escaped}\n</user>"


def get_cors_origins() -> list[str]:
    raw = os.getenv(
        "CORS_ORIGINS",
        os.getenv(
            "FRONTEND_URL",
            "http://localhost:5173,http://127.0.0.1:5173,http://localhost:5174,http://127.0.0.1:5174",
        ),
    )
    origins = [o.strip() for o in raw.split(",") if o.strip()]
    return origins or ["http://localhost:5173"]


# Matches Vite dev servers (localhost, 127.0.0.1, IPv6, LAN IP when host:true)
_DEFAULT_CORS_ORIGIN_REGEX = (
    r"https?://(localhost|127\.0\.0\.1|\[::1\]|192\.168\.\d{1,3}\.\d{1,3})(:\d+)?$"
)


def get_cors_middleware_kwargs() -> dict:
    """
    CORS settings for FastAPI. Uses explicit origins plus a dev regex so HITL
    (e.g. http://localhost:5174 → http://127.0.0.1:8000) preflight succeeds.

    Set CORS_ORIGIN_REGEX=false in production to allow only CORS_ORIGINS.
    """
    kwargs: dict = {
        "allow_origins": get_cors_origins(),
        "allow_credentials": True,
        "allow_methods": ["*"],
        "allow_headers": ["*"],
    }
    regex = os.getenv("CORS_ORIGIN_REGEX", _DEFAULT_CORS_ORIGIN_REGEX).strip()
    if regex.lower() not in {"false", "0", "no", "off", "disabled", ""}:
        kwargs["allow_origin_regex"] = regex
    return kwargs


async def verify_api_key(
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> None:
    """Optional gateway auth — enable with REQUIRE_API_KEY=true and API_KEY set."""
    if not REQUIRE_API_KEY:
        return
    if not API_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API key auth is enabled but API_KEY is not configured.",
        )
    if not x_api_key or x_api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )


def verify_agent_service_key(provided: str | None) -> None:
    """Service-to-service auth for AccuEntry_Agents microservice."""
    expected = AGENT_SERVICE_API_KEY
    if not expected:
        return
    if not provided or provided != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized agent service request.",
        )
