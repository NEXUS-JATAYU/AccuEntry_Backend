"""
OTP Service — generation, hashing, verification, and email dispatch.

All OTPs are SHA-256 hashed before storage. Raw codes are never logged.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import secrets
import time
from typing import Any

import httpx

from core.otp_store import OTPRecord, delete_record, get_record, set_record

logger = logging.getLogger(__name__)

RESEND_API_KEY = os.getenv("RESEND_API_KEY", "re_placeholder_key")
OTP_EXPIRY_SECONDS = 600
MAX_ATTEMPTS = 3
MAX_SENDS_PER_HOUR = 3


def _hash_otp(code: str) -> str:
    return hashlib.sha256(code.encode()).hexdigest()


def mask_email(email: str) -> str:
    if not email or "@" not in email:
        return "****@****.com"
    local, domain = email.split("@", 1)
    visible = local[:2] if len(local) >= 2 else local[:1]
    masked = visible + "****"
    return f"{masked}@{domain}"


def generate_otp(session_id: str) -> str | None:
    now = time.time()
    one_hour_ago = now - 3600

    existing = get_record(session_id)
    recent_sends: list[float] = []
    if existing:
        if (not existing.used) and ((now - existing.created_at) <= OTP_EXPIRY_SECONDS):
            return "ACTIVE_EXISTS"
        recent_sends = [t for t in existing.send_timestamps if t > one_hour_ago]
        if len(recent_sends) >= MAX_SENDS_PER_HOUR:
            logger.warning("OTP rate limit hit for session %s", session_id)
            return None

    code = f"{secrets.randbelow(10000):04d}"
    timestamps = list(recent_sends)
    timestamps.append(now)

    set_record(
        session_id,
        OTPRecord(
            hashed_code=_hash_otp(code),
            created_at=now,
            send_count=len(timestamps),
            send_timestamps=timestamps,
        ),
    )
    return code


def clear_otp(session_id: str) -> None:
    delete_record(session_id)


def verify_otp(session_id: str, submitted_code: str) -> tuple[bool, str]:
    record = get_record(session_id)
    if not record:
        return False, "Invalid code. No activation code was requested for this session."
    if record.used:
        return False, "This code has already been used. Please request a new one."
    if time.time() - record.created_at > OTP_EXPIRY_SECONDS:
        return False, "Your code has expired. Please request a new one."
    if record.attempts >= MAX_ATTEMPTS:
        return False, "Too many incorrect attempts. Please restart the activation process or contact support."

    if _hash_otp(submitted_code) != record.hashed_code:
        record.attempts += 1
        remaining = MAX_ATTEMPTS - record.attempts
        if remaining <= 0:
            record.used = True
            set_record(session_id, record)
            return False, "Too many incorrect attempts. Please restart the activation process or contact support."
        set_record(session_id, record)
        return False, f"Incorrect code. You have {remaining} attempt{'s' if remaining != 1 else ''} remaining."

    record.used = True
    set_record(session_id, record)
    return True, "OTP verified successfully."


def is_otp_locked(session_id: str) -> bool:
    record = get_record(session_id)
    if not record:
        return False
    return record.attempts >= MAX_ATTEMPTS


def otp_recently_sent(session_id: str, window_seconds: int = 180) -> bool:
    record = get_record(session_id)
    if not record or record.used:
        return False
    if time.time() - record.created_at > OTP_EXPIRY_SECONDS:
        return False
    if not record.send_timestamps:
        return False
    return (time.time() - max(record.send_timestamps)) <= max(window_seconds, 1)


def get_otp_send_count(session_id: str) -> int:
    record = get_record(session_id)
    if not record:
        return 0
    return max(0, int(record.send_count))


async def send_otp_email(session_id: str, email: str, otp_code: str) -> bool:
    masked = mask_email(email)
    logger.info("Sending OTP email to %s (session %s)", masked, session_id)

    html_content = f"""
    <div style='font-family:sans-serif;max-width:480px;margin:auto;'>
        <h2 style='color:#1a1a2e;'>Your Activation Code</h2>
        <p>You're almost there! Use the code below to activate your account:</p>
        <div style='font-size:40px;font-weight:bold;letter-spacing:12px;
                    color:#4F46E5;text-align:center;padding:24px;
                    background:#f5f5ff;border-radius:8px;margin:24px 0;'>
            {otp_code}
        </div>
        <p>This code is valid for <strong>10 minutes</strong> and can only be used once.</p>
    </div>
    """

    payload = {
        "from": "AccuEntry <no-reply@accuentry.artistmait.me>",
        "to": [email],
        "subject": "Your Account Activation Code",
        "html": html_content,
    }

    success = await _dispatch_email(payload, session_id, masked)
    if success:
        return True

    logger.warning("Retrying OTP email for %s in 5s...", masked)
    await asyncio.sleep(5)
    retry_success = await _dispatch_email(payload, session_id, masked)
    if not retry_success:
        clear_otp(session_id)
    return retry_success


async def send_confirmation_email(
    session_id: str,
    email: str,
    full_name: str,
    account_id: str,
    account_type: str,
    activation_date: str,
) -> bool:
    masked = mask_email(email)
    html_content = f"""
    <div style='font-family:sans-serif;max-width:480px;margin:auto;'>
        <h2 style='color:#16a34a;'>Account Successfully Activated!</h2>
        <p>Dear {full_name},</p>
        <p>Your account has been verified and is now fully active.</p>
        <div style='background:#f0fdf4;border-radius:8px;padding:20px;margin:20px 0;'>
            <p><strong>Account ID:</strong> {account_id}</p>
            <p><strong>Account Type:</strong> {account_type}</p>
            <p><strong>Activated On:</strong> {activation_date}</p>
        </div>
    </div>
    """
    payload = {
        "from": "AccuEntry <no-reply@accuentry.artistmait.me>",
        "to": [email],
        "subject": "Your Account is Now Active!",
        "html": html_content,
    }
    success = await _dispatch_email(payload, session_id, masked)
    if success:
        return True
    await asyncio.sleep(5)
    return await _dispatch_email(payload, session_id, masked)


async def _dispatch_email(
    payload: dict[str, Any], session_id: str, masked_email: str,
) -> bool:
    if RESEND_API_KEY == "re_placeholder_key":
        logger.info(
            "SIMULATED EMAIL to %s (session %s) — set RESEND_API_KEY for production",
            masked_email,
            session_id,
        )
        return True

    headers = {
        "Authorization": f"Bearer {RESEND_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                "https://api.resend.com/emails",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            logger.info("Email dispatched to %s", masked_email)
            return True
    except httpx.HTTPError as exc:
        logger.error("Email send failed to %s: %s", masked_email, exc)
        return False
    except Exception as exc:
        logger.error("Unexpected email error for %s: %s", masked_email, exc)
        return False
