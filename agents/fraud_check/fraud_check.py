"""
Fraud check subgraph: layered identity validation + LLM risk reasoning.

Architecture
------------
Layer 1  Rule-based velocity & format checks          (fast, deterministic)
Layer 2  Device / behavioural signal scoring           (rule-based)
Layer 3  Identity document cross-matching              (rule-based)
Layer 4  LLM risk reasoning over aggregated signals    (Gemma 2 2B via Ollama)

Open-source tooling (no hardware required)
------------------------------------------
- python-phonenumbers   : phone format + carrier lookup
- pypostal / postal     : address normalisation (optional, install separately)
- email-validator       : RFC-compliant e-mail checks
- FingerprintJS OSS     : browser fingerprint (client-side, passed in state)
- ipsum / plain IP sets : IP blocklist lookups (file-based, no API key)
- Ollama + gemma2:2b    : local LLM risk reasoning (no API key required)

All checks are additive: each layer appends to a `signals` list and
increments a numeric `risk_score`. Layer 4 re-evaluates the full bundle.

Prerequisites for layer 4
--------------------------
1. Install Ollama:  https://ollama.com/download
2. Pull the model:  ollama pull gemma2:2b
3. Start server:    ollama serve   (default: http://localhost:11434)

Environment variables
---------------------
OLLAMA_BASE_URL    : Ollama server URL (default: http://localhost:11434)
FRAUD_LLM_MODEL    : optional model override (default: gemma2:2b)
IP_BLOCKLIST_PATH  : path to newline-separated IP blocklist file (optional)


"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any

import phonenumbers
from email_validator import EmailNotValidError, validate_email
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from memory_manager import AgentMemoryManager
from state import OnboardingState

logger = logging.getLogger(__name__)
_memory = AgentMemoryManager()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
FRAUD_LLM_MODEL = os.getenv("FRAUD_LLM_MODEL", os.getenv("OLLAMA_MODEL", "gemma2:2b"))


def resolve_fraud_action(
    combined_rule_score: int,
    llm_action: str,
    skip_llm_below: int = 25,
) -> tuple[str, int]:
    """
    Map rule score + LLM recommendation to final fraud action.
    Aligns with Layer 4 prompt bands; never auto-clears scores >= 60 unless
    FRAUD_FORCE_CLEAR_BELOW is set (legacy CI / test only).
    """
    force_clear_below = os.getenv("FRAUD_FORCE_CLEAR_BELOW", "").strip()
    if force_clear_below:
        try:
            if combined_rule_score < int(force_clear_below):
                return "clear", combined_rule_score
        except ValueError:
            pass

    action = (
        llm_action
        if llm_action in ("clear", "manual_review", "reject")
        else "manual_review"
    )

    if combined_rule_score < skip_llm_below:
        return "clear", combined_rule_score

    if combined_rule_score >= 75:
        if action == "clear":
            return "reject", combined_rule_score
        return action, combined_rule_score

    if combined_rule_score >= 60:
        if action == "clear":
            return "manual_review", combined_rule_score
        return action, combined_rule_score

    if combined_rule_score >= 50:
        if action == "clear":
            return "manual_review", combined_rule_score
        return action, combined_rule_score

    return action, combined_rule_score


def _document_verified(state: OnboardingState) -> bool:
    if state.get("document_verified") is not None:
        return bool(state.get("document_verified"))
    return bool(state.get("pan_verified")) and bool(state.get("aadhaar_verified"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ollama_generate(prompt: str, timeout: int = 30) -> str:
    """
    POST to Ollama's /api/generate endpoint (non-streaming).
    Returns the model's response text, or raises on failure.
    """
    payload = json.dumps({
        "model": FRAUD_LLM_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,   # low temp for consistent JSON output
            "num_predict": 256,   # enough for our JSON schema, keeps it fast
        },
    }).encode()

    req = urllib.request.Request(
        f"{OLLAMA_BASE_URL}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode())
    return body["response"]


def _load_ip_blocklist() -> set[str]:
    path = os.environ.get("IP_BLOCKLIST_PATH", "")
    if not path or not os.path.exists(path):
        return set()
    try:
        with open(path) as fh:
            return {ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")}
    except OSError:
        logger.warning("Could not read IP blocklist at %s", path)
        return set()


_IP_BLOCKLIST: set[str] = _load_ip_blocklist()


def _store_fraud_memory(
    state: OnboardingState,
    *,
    l1_score: int,
    l1_signals: list[str],
    l2_score: int,
    l2_signals: list[str],
    l3_score: int,
    l3_signals: list[str],
    llm_result: dict[str, Any],
    action: str,
    outcome_stage: str,
    fraud_status: str,
    risk_score: int,
) -> None:
    session_id = state.get("audit_session_id") or state.get("session_id") or "unknown"
    _memory.store_interaction(
        session_id=session_id,
        agent_name="fraud_check",
        input_data={
            "layer1": {"score": l1_score, "signals": l1_signals},
            "layer2": {"score": l2_score, "signals": l2_signals},
            "layer3": {"score": l3_score, "signals": l3_signals},
            "combined_rule_score": l1_score + l2_score + l3_score,
            "llm_result": llm_result,
            "aml_status": state.get("aml_status"),
        },
        output_data={
            "action": action,
            "fraud_status": fraud_status,
            "fraud_risk_score": risk_score,
            "outcome_stage": outcome_stage,
            "reasoning": llm_result.get("reasoning") or "",
        },
        risk_score=float(risk_score),
        decision=action,
        metadata={
            "audit_session_id": state.get("audit_session_id") or session_id,
            "workflow_stage": "fraud_check",
            "outcome_stage": outcome_stage,
            "risk_level": llm_result.get("risk_level", "unknown"),
            "fraud_status": fraud_status,
        },
        event_type="fraud_outcome",
    )


# ---------------------------------------------------------------------------
# Layer 1: Rule-based velocity & format checks
# ---------------------------------------------------------------------------

def _check_email_format(email: str | None, signals: list[str]) -> int:
    """Return risk delta (0 = clean, positive = suspicious)."""
    if not email:
        signals.append("missing_email")
        return 20
    try:
        validate_email(email, check_deliverability=False)
    except EmailNotValidError as exc:
        signals.append(f"invalid_email_format:{exc}")
        return 25
    disposable_domains = {"mailinator.com", "guerrillamail.com", "tempmail.com", "throwam.com"}
    domain = email.split("@")[-1].lower()
    if domain in disposable_domains:
        signals.append(f"disposable_email_domain:{domain}")
        return 30
    return 0


def _check_phone_format(phone: str | None, signals: list[str]) -> int:
    if not phone:
        signals.append("missing_phone")
        return 15
    try:
        parsed = phonenumbers.parse(phone, None)
        if not phonenumbers.is_valid_number(parsed):
            signals.append("invalid_phone_number")
            return 20
        number_type = phonenumbers.number_type(parsed)
        # VOIP / unknown number types are higher risk
        if number_type in (
            phonenumbers.PhoneNumberType.VOIP,
            phonenumbers.PhoneNumberType.UNKNOWN,
        ):
            signals.append(f"risky_phone_type:{number_type.name}")
            return 20
    except phonenumbers.NumberParseException:
        signals.append("unparseable_phone")
        return 25
    return 0


def _check_ip(ip: str | None, signals: list[str]) -> int:
    if not ip:
        return 0
    if ip in _IP_BLOCKLIST:
        signals.append(f"ip_blocklisted:{ip}")
        return 40
    # Simple private/loopback check â€” not a risk signal, but flag for testing
    if ip.startswith(("10.", "192.168.", "127.")):
        signals.append("ip_private_range")
    return 0


def _check_signup_velocity(state: OnboardingState, signals: list[str]) -> int:
    """
    Check how many accounts have been created from this IP recently.
    Expects state to carry an optional 'ip_signup_count_24h' int set by an
    upstream middleware / rate-limiter.
    """
    count = state.get("ip_signup_count_24h", 0)
    if count >= 5:
        signals.append(f"high_velocity_ip:{count}_in_24h")
        return 35
    if count >= 3:
        signals.append(f"moderate_velocity_ip:{count}_in_24h")
        return 15
    return 0


def layer1_rule_checks(state: OnboardingState) -> tuple[int, list[str]]:
    signals: list[str] = []
    score = 0
    score += _check_email_format(state.get("email_id"), signals)
    score += _check_phone_format(state.get("mobile_number"), signals)
    score += _check_ip(state.get("ip_address"), signals)
    score += _check_signup_velocity(state, signals)
    return score, signals


# ---------------------------------------------------------------------------
# Layer 2: Device & behavioural signals
# ---------------------------------------------------------------------------

def layer2_behavioural(state: OnboardingState) -> tuple[int, list[str]]:
    """
    Expects the client to pass these optional fields in state:
      fingerprint_id      : str   — FingerprintJS visitor ID
      fingerprint_flags   : list  — e.g. ["headless", "vpn", "tor"]
      form_fill_seconds   : float — total time from page load to submission
      keystroke_entropy   : float — 0.0 (robotic) … 1.0 (natural)
      recaptcha_score     : float — Google reCAPTCHA v3 score (0.0 to 1.0, where 0.0 is bot)
      typing_velocity     : float — Keystrokes per second (cps)
    """
    signals: list[str] = []
    score = 0

    flags: list[str] = state.get("fingerprint_flags") or []
    for flag in flags:
        flag_lower = flag.lower()
        if flag_lower in ("headless", "bot"):
            signals.append(f"device_flag:{flag_lower}")
            score += 40
        elif flag_lower in ("vpn", "proxy"):
            signals.append(f"device_flag:{flag_lower}")
            score += 15
        elif flag_lower == "tor":
            signals.append("device_flag:tor")
            score += 30

    fill_seconds: float = state.get("form_fill_seconds", -1.0)
    if 0 < fill_seconds < 2.0:
        signals.append(f"suspiciously_fast_form_fill:{fill_seconds:.1f}s")
        score += 25

    entropy: float = state.get("keystroke_entropy", -1.0)
    if 0.0 <= entropy < 0.2:
        signals.append(f"low_keystroke_entropy:{entropy:.2f}")
        score += 20
        
    recaptcha_score: float = state.get("recaptcha_score", -1.0)
    if 0.0 <= recaptcha_score <= 0.3:
        signals.append(f"low_recaptcha_score:{recaptcha_score:.2f}")
        score += 35
    elif 0.3 < recaptcha_score < 0.6:
        signals.append(f"medium_recaptcha_score:{recaptcha_score:.2f}")
        score += 15

    typing_velocity: float = state.get("typing_velocity", -1.0)
    if typing_velocity > 15.0:
        signals.append(f"high_typing_velocity_bot:{typing_velocity:.1f}cps")
        score += 30
    elif typing_velocity > 10.0:
        signals.append(f"fast_typing_velocity:{typing_velocity:.1f}cps")
        score += 10

    return score, signals


# ---------------------------------------------------------------------------
# Layer 3: Identity document cross-matching
# ---------------------------------------------------------------------------

def _dob_matches(dob_str: str | None, document_dob: str | None, signals: list[str]) -> int:
    if not dob_str or not document_dob:
        return 0
    try:
        claimed = datetime.strptime(dob_str, "%Y-%m-%d").date()
        doc = datetime.strptime(document_dob, "%Y-%m-%d").date()
        if claimed != doc:
            signals.append("dob_mismatch_vs_document")
            return 35
    except ValueError:
        signals.append("dob_unparseable")
        return 10
    return 0


def _name_similarity(a: str, b: str) -> float:
    """Crude Jaccard similarity on character trigrams â€” no external library."""
    def trigrams(s: str) -> set[str]:
        s = re.sub(r"\s+", "", s.lower())
        return {s[i : i + 3] for i in range(len(s) - 2)} if len(s) >= 3 else {s}

    ta, tb = trigrams(a), trigrams(b)
    if not ta and not tb:
        return 1.0
    return len(ta & tb) / len(ta | tb)


def layer3_identity(state: OnboardingState) -> tuple[int, list[str]]:
    """
    Expects optional state fields:
      full_name           : str  â€” user-submitted name
      document_name       : str  â€” name extracted from ID document
      date_of_birth       : str  â€” user-submitted (YYYY-MM-DD)
      document_dob        : str  â€” DOB extracted from ID document
      address             : str  â€” user-submitted address string
      document_address    : str  â€” address on ID document
    """
    signals: list[str] = []
    score = 0

    name: str = state.get("full_name") or ""
    doc_name: str = state.get("document_name") or ""
    if name and doc_name:
        sim = _name_similarity(name, doc_name)
        if sim < 0.4:
            signals.append(f"name_low_similarity:{sim:.2f}")
            score += 30
        elif sim < 0.7:
            signals.append(f"name_partial_similarity:{sim:.2f}")
            score += 10

    user_dob = state.get("dob") or state.get("date_of_birth")
    score += _dob_matches(
        user_dob,
        state.get("document_dob"),
        signals,
    )

    # Basic address cross-check (token overlap, libpostal optional)
    addr: str = (state.get("address") or "").lower()
    doc_addr: str = (state.get("document_address") or "").lower()
    if addr and doc_addr:
        addr_tokens = set(re.findall(r"\w+", addr))
        doc_tokens = set(re.findall(r"\w+", doc_addr))
        overlap = len(addr_tokens & doc_tokens) / max(len(addr_tokens | doc_tokens), 1)
        if overlap < 0.3:
            signals.append(f"address_low_overlap:{overlap:.2f}")
            score += 20

    return score, signals


# ---------------------------------------------------------------------------
# Layer 4: LLM risk reasoning
# ---------------------------------------------------------------------------

# Few-shot prompt works better than a system prompt for small models.
# One complete example teaches the exact JSON shape we expect.
_PROMPT_TEMPLATE = """You are a fraud-risk analyst. Analyse the signals below and reply with ONLY a JSON object â€” no explanation, no markdown fences.

Required JSON schema:
{{"risk_score": <int 0-100>, "risk_level": "<low|medium|high|critical>", "recommended_action": "<clear|manual_review|reject>", "reasoning": "<max 60 words>", "top_signals": [<up to 3 signal strings>]}}

Scoring guide: 0-24=low/clear, 25-49=medium/clear, 50-74=high/manual_review, 75-100=critical/reject. Prefer manual_review over reject when uncertain.

Example input:
{{"rule_based_score": 55, "signals": ["disposable_email_domain:mailinator.com", "device_flag:vpn"], "document_verified": false}}
Example output:
{{"risk_score": 60, "risk_level": "high", "recommended_action": "manual_review", "reasoning": "Disposable email and VPN usage without document verification warrant human review.", "top_signals": ["disposable_email_domain:mailinator.com", "device_flag:vpn"]}}

Now analyse this input and reply with ONLY the JSON object:
{bundle}"""


async def layer4_llm_reasoning(
    rule_score: int,
    signals: list[str],
    state: OnboardingState,
) -> dict[str, Any]:
    """Call Gemma 2 2B via Ollama to synthesise signals into a risk decision."""
    email = state.get("email_id") or state.get("email") or ""
    phone = state.get("mobile_number") or state.get("phone")

    bundle: dict[str, Any] = {
        "rule_based_score": rule_score,
        "signals": signals,
        "email_domain": email.split("@")[-1] if "@" in email else None,
        "phone_country": None,
        "ip_country": state.get("ip_country"),
        "aml_status": state.get("aml_status"),
        "form_fill_seconds": state.get("form_fill_seconds"),
        "keystroke_entropy": state.get("keystroke_entropy"),
        "document_verified": _document_verified(state),
    }

    if phone:
        try:
            parsed = phonenumbers.parse(phone, None)
            bundle["phone_country"] = phonenumbers.region_code_for_number(parsed)
        except phonenumbers.NumberParseException:
            pass

    def _fallback(reason: str) -> dict[str, Any]:
        logger.warning("LLM risk reasoning failed (%s), falling back to rule score", reason)
        if rule_score >= 75:
            action, level = "reject", "critical"
        elif rule_score >= 50:
            action, level = "manual_review", "high"
        elif rule_score >= 25:
            action, level = "manual_review", "medium"
        else:
            action, level = "clear", "low"
        return {
            "risk_score": rule_score,
            "risk_level": level,
            "recommended_action": action,
            "reasoning": f"LLM unavailable ({reason}). Rule score: {rule_score}. Signals: {', '.join(signals) or 'none'}.",
            "top_signals": signals[:3],
        }

    try:
        prompt = _PROMPT_TEMPLATE.format(bundle=json.dumps(bundle))
        raw = await asyncio.to_thread(_ollama_generate, prompt)

        # gemma2:2b sometimes wraps output in ```json ... ``` â€” strip it
        raw = re.sub(r"^```json\s*|\s*```$", "", raw.strip(), flags=re.DOTALL).strip()

        # Extract first {...} block in case the model adds trailing commentary
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return _fallback("no JSON block in response")
        result: dict[str, Any] = json.loads(match.group())

        # Validate required keys are present and sensible
        action = result.get("recommended_action", "")
        if action not in ("clear", "manual_review", "reject"):
            return _fallback(f"invalid recommended_action: {action!r}")

        return result

    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            reason = (
                f"Ollama model not found (HTTP 404) for '{FRAUD_LLM_MODEL}' at {OLLAMA_BASE_URL}. "
                f"Run: ollama pull {FRAUD_LLM_MODEL}"
            )
        else:
            reason = f"Ollama HTTP error {exc.code}: {exc.reason}"
        return _fallback(reason)
    except urllib.error.URLError as exc:
        return _fallback(f"Ollama unreachable: {exc}")
    except json.JSONDecodeError as exc:
        return _fallback(f"JSON parse error: {exc}")
    except Exception as exc:
        return _fallback(str(exc))


# ---------------------------------------------------------------------------
# Main check node
# ---------------------------------------------------------------------------

async def check_node(state: OnboardingState) -> dict[str, Any]:
    # â”€â”€ AML gate (unchanged logic) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    aml_status = state.get("aml_status")
    aml_in_background = bool(state.get("aml_in_background"))

    # Debug logging for data capture fields
    logger.info(
        "fraud_check.check_node: email_id=%s mobile_number=%s full_name=%s",
        state.get("email_id"),
        state.get("mobile_number"),
        state.get("full_name"),
    )

    if aml_in_background or aml_status in (None, "pending", "checking"):
        # If AML has already completed, proceed regardless of stale flags.
        if not state.get("aml_completed"):
            # AML is still running; avoid heavy LLM fraud checks to keep chat responsive.
            return {
                "stage": "fraud_check",
                "fraud_status": "pending_aml",
                "fraud_risk_score": state.get("fraud_risk_score") or 0,
                "fraud_signals": state.get("fraud_signals") or [],
                "progress": max(state.get("progress", 0), 80),
                "messages": [
                    {
                        "role": "assistant",
                        "text": "AML screening is in progress. We will continue immediately once checks are complete.",
                    }
                ],
            }

    if aml_status == "review":
        _store_fraud_memory(
            state,
            l1_score=0,
            l1_signals=[],
            l2_score=0,
            l2_signals=[],
            l3_score=0,
            l3_signals=[],
            llm_result={"recommended_action": "manual_review", "reasoning": "AML review required"},
            action="manual_review",
            outcome_stage="manual_review",
            fraud_status="pending_aml_review",
            risk_score=0,
        )
        return {
            "stage": "manual_review",
            "fraud_status": "pending_aml_review",
            "fraud_risk_score": state.get("fraud_risk_score") or 0,
            "fraud_signals": state.get("fraud_signals") or [],
            "progress": 85,
            "messages": [
                {
                    "role": "assistant",
                    "text": "AML screening requires manual compliance review before we can continue.",
                }
            ],
        }

    if aml_status == "flagged":
        _memory.store_interaction(
            session_id=state.get("audit_session_id") or state.get("session_id") or "unknown",
            agent_name="fraud_check",
            input_data={
                "aml_status": aml_status,
                "fraud_risk_score": state.get("fraud_risk_score") or 0,
                "fraud_signals": state.get("fraud_signals") or [],
            },
            output_data={
                "action": "reject",
                "fraud_status": "flagged",
                "outcome_stage": "rejected",
            },
            risk_score=float(state.get("fraud_risk_score") or 0),
            decision="reject",
            metadata={
                "audit_session_id": state.get("audit_session_id") or state.get("session_id") or "unknown",
                "workflow_stage": "fraud_check",
                "outcome_stage": "rejected",
                "fraud_status": "flagged",
            },
            event_type="fraud_outcome",
        )
        from agents.aml.aml_user_report import build_aml_flag_user_message

        report_text = await build_aml_flag_user_message(state)
        return {
            "stage": "rejected",
            "fraud_status": "flagged",
            "fraud_risk_score": state.get("fraud_risk_score") or 0,
            "fraud_signals": state.get("fraud_signals") or [],
            "progress": 100,
            "decision_action": "reject",
            "decision_reason": "AML flagged during fraud screening gate",
            "assigned_employee_name": state.get("assigned_employee_name"),
            "assigned_bank_branch": state.get("assigned_bank_branch"),
            "assigned_date": state.get("assigned_date"),
            "assigned_time": state.get("assigned_time"),
            "messages": [{"role": "assistant", "text": report_text}],
        }

    # NOTE: We no longer bypass the decision agent if fraud is cleared.
    # We must proceed to the decision agent so OTP activation can trigger.

    # â”€â”€ Layered fraud checks â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    t0 = time.monotonic()

    l1_score, l1_signals = layer1_rule_checks(state)
    l2_score, l2_signals = layer2_behavioural(state)
    l3_score, l3_signals = layer3_identity(state)

    combined_rule_score = l1_score + l2_score + l3_score
    all_signals = l1_signals + l2_signals + l3_signals

    skip_llm_below = int(os.getenv("FRAUD_SKIP_LLM_BELOW", "25"))
    if combined_rule_score < skip_llm_below:
        llm_result = {
            "risk_score": combined_rule_score,
            "risk_level": "low",
            "recommended_action": "clear",
            "reasoning": f"Rule score {combined_rule_score} below LLM threshold; skipped Ollama.",
            "top_signals": all_signals[:3],
        }
    else:
        llm_result = await layer4_llm_reasoning(combined_rule_score, all_signals, state)
    elapsed = time.monotonic() - t0

    action: str = llm_result.get("recommended_action", "manual_review")
    risk_level: str = llm_result.get("risk_level", "medium")
    reasoning: str = llm_result.get("reasoning", "")
    risk_score: int = llm_result.get("risk_score", combined_rule_score)

    logger.info(
        "fraud_check action=%s risk=%s score=%d elapsed=%.2fs signals=%s",
        action,
        risk_level,
        risk_score,
        elapsed,
        all_signals,
    )

    action, risk_score = resolve_fraud_action(
        combined_rule_score,
        action,
        skip_llm_below=skip_llm_below,
    )

    # Route on resolved action
    if action == "reject":
        _store_fraud_memory(
            state,
            l1_score=l1_score,
            l1_signals=l1_signals,
            l2_score=l2_score,
            l2_signals=l2_signals,
            l3_score=l3_score,
            l3_signals=l3_signals,
            llm_result=llm_result,
            action="reject",
            outcome_stage="rejected",
            fraud_status="rejected",
            risk_score=int(risk_score),
        )
        return {
            "stage": "rejected",
            "fraud_status": "rejected",
            "fraud_risk_score": risk_score,
            "fraud_signals": all_signals,
            "progress": 100,
            "messages": [
                {
                    "role": "assistant",
                    "text": "We are unable to proceed with this application at this time.",
                }
            ],
        }

    if action == "manual_review":
        _store_fraud_memory(
            state,
            l1_score=l1_score,
            l1_signals=l1_signals,
            l2_score=l2_score,
            l2_signals=l2_signals,
            l3_score=l3_score,
            l3_signals=l3_signals,
            llm_result=llm_result,
            action="manual_review",
            outcome_stage="manual_review",
            fraud_status="review",
            risk_score=int(risk_score),
        )
        return {
            "stage": "manual_review",
            "fraud_status": "review",
            "fraud_risk_score": risk_score,
            "fraud_signals": all_signals,
            "fraud_reasoning": reasoning,
            "progress": 92,
            "messages": [
                {
                    "role": "assistant",
                    "text": (
                        "Your application requires a brief additional review. "
                        "Our team will be in touch within 1 business day."
                    ),
                }
            ],
        }

    # action == "clear"
    # Transition to decision_agent stage so the final decision
    # agent can trigger the OTP email via the approve_account tool.
    _store_fraud_memory(
        state,
        l1_score=l1_score,
        l1_signals=l1_signals,
        l2_score=l2_score,
        l2_signals=l2_signals,
        l3_score=l3_score,
        l3_signals=l3_signals,
        llm_result=llm_result,
        action="clear",
        outcome_stage="decision_agent",
        fraud_status="clear",
        risk_score=int(risk_score),
    )
    return {
        "fraud_status": "clear",
        "fraud_risk_score": risk_score,
        "fraud_signals": all_signals,
        "fraud_reasoning": reasoning,
        "stage": "decision_agent",
        "progress": 95,
        "messages": [
            {
                "role": "assistant",
                "text": "Fraud checks cleared. Finalizing your account activation...",
            }
        ],
    }


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def build_fraud_check_graph() -> CompiledStateGraph:
    workflow = StateGraph(OnboardingState)
    workflow.add_node("check_node", check_node)
    workflow.add_edge(START, "check_node")
    workflow.add_edge("check_node", END)
    return workflow.compile()
