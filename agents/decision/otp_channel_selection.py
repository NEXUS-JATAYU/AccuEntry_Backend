"""
OTP Channel Selection Node: Handle user preference for Email or Phone OTP.

This node is invoked when stage="otp_channel_selection".
It expects the user to provide their channel choice ("email" or "phone").
Based on the choice, it generates an OTP and sends it via the chosen channel.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from agents.decision.otp_service import (
    clear_otp,
    generate_otp,
    send_otp_email,
    send_otp_sms,
    mask_email,
    otp_recently_sent,
)
from state import OnboardingState

logger = logging.getLogger(__name__)


async def otp_channel_selection_node(state: OnboardingState) -> dict[str, Any]:
    """
    Handle OTP channel selection from user input.
    """
    session_id = state.get("audit_session_id") or state.get("session_id", "")
    messages = state.get("messages", [])

    if not messages:
        return _prompt_selection()

    last_message = None
    for msg in reversed(messages):
        if msg.get("role") == "user":
            last_message = msg
            break

    if not last_message:
        return _prompt_selection()

    user_text = last_message.get("text", "").strip().lower()
    
    channel = None
    if "email" in user_text:
        channel = "email"
    elif "phone" in user_text or "sms" in user_text or "mobile" in user_text:
        channel = "phone"

    if not channel:
        return _prompt_selection(error="Please clearly specify 'Email' or 'Phone'.")

    # Generate and send OTP
    email_id = state.get("email_id") or ""
    phone_number = state.get("mobile_number") or ""
    masked_email = mask_email(email_id)

    if otp_recently_sent(session_id, window_seconds=180):
        # Already sent recently
        return _verification_state(
            channel,
            masked_email,
            phone_number,
            "A code was already sent recently. Please enter it, or type 'resend code' if needed."
        )

    otp_code = generate_otp(session_id)
    
    if otp_code == "ACTIVE_EXISTS":
        return _verification_state(
            channel,
            masked_email,
            phone_number,
            "An activation code is already active. Please enter it, or type 'resend code' if needed."
        )
    elif otp_code is None:
        return _verification_state(
            channel,
            masked_email,
            phone_number,
            "We're unable to send a new activation code right now due to limits. Please wait a few minutes and type 'resend code'."
        )

    # Send via selected channel
    sent = False
    if channel == "email":
        sent = await send_otp_email(session_id, email_id, otp_code)
    else:
        sent = await send_otp_sms(session_id, phone_number, otp_code)

    if sent:
        return _verification_state(
            channel,
            masked_email,
            phone_number,
            "We've reached the final stage of setting up your account!\n"
            f"We have sent a 4-digit Activation Code to your {'email ' + masked_email if channel == 'email' else 'phone number ' + phone_number}.\n"
            "Please enter the code to activate your account."
        )
    else:
        clear_otp(session_id)
        return _verification_state(
            channel,
            masked_email,
            phone_number,
            f"We could not deliver your activation code to your {channel}. Please type 'resend code' to try again."
        )


def _prompt_selection(error: str = "") -> dict[str, Any]:
    msg = "How would you like to receive your activation code: Email or Phone?"
    if error:
        msg = f"{error}\n{msg}"
        
    return {
        "stage": "otp_channel_selection",
        "messages": [
            {
                "role": "assistant",
                "type": "OTP_CHANNEL_REQUESTED",
                "payload": {
                    "message": msg
                }
            }
        ]
    }


def _verification_state(channel: str, masked_email: str, phone: str, text: str) -> dict[str, Any]:
    return {
        "stage": "otp_verification",
        "messages": [
            {
                "role": "assistant",
                "type": "OTP_REQUESTED",
                "payload": {
                    "message": text,
                    "inputType": "otp",
                    "otpLength": 4,
                    "expiresInMinutes": 10,
                    "channel": channel
                }
            }
        ]
    }
