"""
Account Activation Service

Generates secure, time-limited JWT activation links and dispatches
activation emails using the Resend API. Includes retry and error
logging mechanisms to ensure robustness inside the decision agent.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any

import httpx
import jwt
import time

logger = logging.getLogger(__name__)

# Temporary in-memory store for rate limiting (session_id -> list of timestamps)
_rate_limits: dict[str, list[float]] = {}

def _check_rate_limit(session_id: str) -> bool:
    """Returns True if within rate limit (max 3 per hour), False if exceeded."""
    now = time.time()
    one_hour_ago = now - 3600
    
    limits = _rate_limits.get(session_id, [])
    limits = [t for t in limits if t > one_hour_ago]
    
    if len(limits) >= 3:
        _rate_limits[session_id] = limits
        return False
        
    limits.append(now)
    _rate_limits[session_id] = limits
    return True

RESEND_API_KEY = os.getenv("RESEND_API_KEY", "re_placeholder_key")
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "fallback-secret-for-jwt-do-not-use-in-prod")
APP_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")


def _generate_activation_token(session_id: str, account_id: str) -> str:
    """Generates a signed, single-use, 24-hour expiration JWT."""
    now = datetime.now(timezone.utc)
    payload = {
        "sub": session_id,
        "acc": account_id,
        "iat": now,
        "exp": now + timedelta(hours=24),
        "jti": str(uuid.uuid4()),  # unique token ID to prevent replay
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm="HS256")


async def send_activation_email(
    session_id: str, email: str, full_name: str, account_id: str, account_type: str
) -> bool:
    """
    Sends the activation email via Resend API.
    Retries once after 5 seconds on failure. Returns True on success, False on failure.
    """
    if not email or "@" not in email:
        logger.error(f"Cannot send activation email for session {session_id}: Invalid email")
        return False
        
    if not _check_rate_limit(session_id):
        logger.error(f"Rate limit exceeded (max 3/hr) for session {session_id}")
        return False
        
    masked_email = f"{email[0]}***@{email.split('@')[1]}"
    logger.info(f"Preparing activation email for {masked_email} (Account: {account_id})")

    token = _generate_activation_token(session_id, account_id)
    activation_link = f"{APP_URL}/activate?token={token}"

    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    html_content = f"""
      <h2>Welcome, {full_name}!</h2>
      <p>Your account has been successfully verified and activated.</p>

      <h3>Account Details</h3>
      <ul>
        <li><strong>Account ID:</strong> {account_id}</li>
        <li><strong>Account Type:</strong> {account_type}</li>
        <li><strong>Activated On:</strong> {now_iso}</li>
        <li><strong>Status:</strong> Active</li>
      </ul>

      <p>Click the button below to log in and get started:</p>
      <a href='{activation_link}' 
         style='background:#4F46E5;color:white;padding:12px 24px;
                border-radius:6px;text-decoration:none;display:inline-block;'>
        Activate & Login →
      </a>

      <p style='color:gray;font-size:12px;margin-top:24px;'>
        This link expires in 24 hours. If you did not request this, 
        please contact support immediately.
      </p>
    """

    payload = {
        "from": "AccuEntry <no-reply@accuentry.artistmait.me>",
        "to": [email],
        "subject": "Your Account is Now Active 🎉",
        "html": html_content,
    }

    headers = {
        "Authorization": f"Bearer {RESEND_API_KEY}",
        "Content-Type": "application/json",
    }

    # Attempt 1
    success = await _send_resend(payload, headers, session_id, masked_email)
    if success:
        return True

    # Retry Once after 5 seconds
    logger.warning(f"Retrying email for {masked_email} in 5s... (Session: {session_id})")
    await asyncio.sleep(5)
    return await _send_resend(payload, headers, session_id, masked_email)


async def _send_resend(
    payload: dict[str, Any], headers: dict[str, str], session_id: str, masked_email: str
) -> bool:
    """Internal helper to dispatch the HTTP request."""
    # In a local test environment, if the token is the placeholder, just simulate success.
    if RESEND_API_KEY == "re_placeholder_key":
        logger.info(f"SIMULATED EMAIL SUCCESS to {masked_email} (Session {session_id}) "
                    f"- Provide a real RESEND_API_KEY environment variable to dispatch.")
        return True

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                "https://api.resend.com/emails",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            logger.info(f"Activation email dispatched to {masked_email}")
            return True
    except httpx.HTTPError as exc:
        logger.error(f"Failed to send email to {masked_email} (Session {session_id}): {str(exc)}")
        return False
    except Exception as exc:
        logger.error(f"Unexpected error sending email to {masked_email} (Session {session_id}): {str(exc)}")
        return False
