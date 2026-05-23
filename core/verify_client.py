"""
AccuVerify HTTP auth headers for Cloud Run (IAM) + service API key.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_ACCUVERIFY_URL = os.getenv("ACCUVERIFY_URL", "http://127.0.0.1:9000").rstrip("/")
_USE_IAM = os.getenv("ACCUVERIFY_USE_IAM", "false").lower() in {"1", "true", "yes"}
_VERIFY_KEY = os.getenv("VERIFY_SERVICE_API_KEY", "").strip()
_ID_TOKEN: str | None = None


def _fetch_id_token(audience: str) -> str | None:
    """GCP identity token for Cloud Run service-to-service calls."""
    try:
        import google.auth.transport.requests
        import google.oauth2.id_token
    except ImportError:
        logger.warning("google-auth not installed; skipping IAM token for AccuVerify")
        return None

    try:
        request = google.auth.transport.requests.Request()
        return google.oauth2.id_token.fetch_id_token(request, audience)
    except Exception as exc:
        logger.warning("Failed to fetch GCP ID token for %s: %s", audience, exc)
        return None


def get_verify_request_headers() -> dict[str, str]:
    """Headers for Backend → AccuVerify requests."""
    global _ID_TOKEN
    headers: dict[str, str] = {}

    if _VERIFY_KEY:
        headers["X-Verify-Service-Key"] = _VERIFY_KEY

    if _USE_IAM and _ACCUVERIFY_URL.startswith("https://"):
        if _ID_TOKEN is None:
            _ID_TOKEN = _fetch_id_token(_ACCUVERIFY_URL)
        if _ID_TOKEN:
            headers["Authorization"] = f"Bearer {_ID_TOKEN}"

    return headers
