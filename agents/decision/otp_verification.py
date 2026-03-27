"""
OTP Verification Node: Handle user OTP submission and verification.

This node is invoked when stage="otp_verification" and receives the user's
OTP submission from the chat. It verifies the code and routes to either:
- stage="complete" on success (proceeds to account activation)
- stage="otp_verification" on invalid code (user retries)
- stage="rejected" on expiry or max attempts reached
"""

from __future__ import annotations

import json
import logging
from typing import Any

from agents.decision.otp_service import verify_otp
from state import OnboardingState

logger = logging.getLogger(__name__)


def otp_verification_node(state: OnboardingState) -> dict[str, Any]:
    """
    Handle OTP verification from user input.

    Expected message format:
    - Regular message with just 4 digits: "1234"
    - Or structured JSON with OTP_SUBMITTED type

    Returns updated state with:
    - stage="complete" on success → triggers downstream completion
    - stage="otp_verification" on invalid (user retries)
    - stage="rejected" on expired (new OTP request needed)
    """
    session_id = state.get("audit_session_id") or state.get("session_id", "")
    messages = state.get("messages", [])

    if not messages:
        logger.warning("otp_verification_node: no messages in state")
        return {
            "messages": [
                {
                    "role": "assistant",
                    "text": "System error: no input received. Please try again.",
                }
            ],
        }

    # Get the last user message
    last_message = None
    for msg in reversed(messages):
        if msg.get("role") == "user":
            last_message = msg
            break

    if not last_message:
        logger.warning("otp_verification_node: no user message found")
        return {
            "messages": [
                {
                    "role": "assistant",
                    "text": "System error: no input received. Please try again.",
                }
            ],
        }

    # Extract OTP code from user input
    user_text = last_message.get("text", "").strip()

    # Try to parse as structured JSON (if frontend sends OTP_SUBMITTED)
    otp_code = None
    try:
        data = json.loads(user_text)
        if isinstance(data, dict) and data.get("type") == "OTP_SUBMITTED":
            otp_code = data.get("payload", {}).get("code", "")
    except (json.JSONDecodeError, ValueError):
        # Not JSON, treat as raw input
        pass

    # If not JSON, extract 4 digits from raw text
    if not otp_code:
        # Remove whitespace and take first 4 digits
        clean_text = "".join(c for c in user_text if c.isdigit())
        otp_code = clean_text[:4]

    if not otp_code or len(otp_code) != 4:
        logger.warning(
            "otp_verification_node: invalid OTP format from user, len=%d",
            len(otp_code),
        )
        return {
            "stage": "otp_verification",
            "messages": [
                {
                    "role": "assistant",
                    "text": (
                        "Please enter the 4-digit code we sent to your email. "
                        "Make sure to enter only the digits."
                    ),
                }
            ],
        }

    # Verify the OTP
    verified, message = verify_otp(session_id, otp_code)

    if verified:
        # Success! Account is now activated
        logger.info(
            "otp_verification_node: OTP verified for session=%s", session_id
        )
        return {
            "stage": "complete",
            "otp_verified": True,
            "progress": 100,
            "messages": [
                {
                    "role": "assistant",
                    "text": (
                        "Congratulations! 🎉\n"
                        "Your Account has been Activated!\n"
                        "Thank You For Banking With Us!"
                    ),
                }
            ],
        }

    # If message indicates expiry or max attempts, reject the application
    if "expired" in message.lower() or "max attempts" in message.lower():
        logger.info(
            "otp_verification_node: OTP rejected for session=%s, reason=%s",
            session_id,
            message,
        )
        return {
            "stage": "rejected",
            "messages": [
                {
                    "role": "assistant",
                    "text": (
                        f"We could not verify your code.\n{message}\n"
                        "Please contact support to request a new activation code."
                    ),
                }
            ],
        }

    # Invalid code, prompt retry (stay in otp_verification stage)
    logger.info(
        "otp_verification_node: invalid OTP for session=%s, attempt count incremented",
        session_id,
    )
    return {
        "stage": "otp_verification",
        "messages": [
            {
                "role": "assistant",
                "text": (
                    f"{message}\n"
                    "Please try again. You have limited attempts remaining."
                ),
            }
        ],
    }
