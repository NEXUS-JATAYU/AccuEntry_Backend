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

NOTE: gemma2:2b is a small model — JSON instruction-following can be
inconsistent. The prompt uses a few-shot example and the parser is
defensive. If the model drifts, the rule-based fallback takes over
automatically so onboarding is never blocked.
"""

from __future__ import annotations

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

from state import OnboardingState

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = "http://localhost:11434"
FRAUD_LLM_MODEL = "gemma2:2b"
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
    # Simple private/loopback check — not a risk signal, but flag for testing
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
    score += _check_email_format(state.get("email"), signals)
    score += _check_phone_format(state.get("phone"), signals)
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
    """Crude Jaccard similarity on character trigrams — no external library."""
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
      full_name           : str  — user-submitted name
      document_name       : str  — name extracted from ID document
      date_of_birth       : str  — user-submitted (YYYY-MM-DD)
      document_dob        : str  — DOB extracted from ID document
      address             : str  — user-submitted address string
      document_address    : str  — address on ID document
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

    score += _dob_matches(
        state.get("date_of_birth"),
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
_PROMPT_TEMPLATE = """You are a fraud-risk analyst. Analyse the signals below and reply with ONLY a JSON object — no explanation, no markdown fences.

Required JSON schema:
{{"risk_score": <int 0-100>, "risk_level": "<low|medium|high|critical>", "recommended_action": "<clear|manual_review|reject>", "reasoning": "<max 60 words>", "top_signals": [<up to 3 signal strings>]}}

Scoring guide: 0-24=low/clear, 25-49=medium/clear, 50-74=high/manual_review, 75-100=critical/reject. Prefer manual_review over reject when uncertain.

Example input:
{{"rule_based_score": 55, "signals": ["disposable_email_domain:mailinator.com", "device_flag:vpn"], "document_verified": false}}
Example output:
{{"risk_score": 60, "risk_level": "high", "recommended_action": "manual_review", "reasoning": "Disposable email and VPN usage without document verification warrant human review.", "top_signals": ["disposable_email_domain:mailinator.com", "device_flag:vpn"]}}

Now analyse this input and reply with ONLY the JSON object:
{bundle}"""


def layer4_llm_reasoning(
    rule_score: int,
    signals: list[str],
    state: OnboardingState,
) -> dict[str, Any]:
    """Call Gemma 2 2B via Ollama to synthesise signals into a risk decision."""
    bundle: dict[str, Any] = {
        "rule_based_score": rule_score,
        "signals": signals,
        "email_domain": (state.get("email") or "").split("@")[-1],
        "phone_country": None,
        "ip_country": state.get("ip_country"),
        "aml_status": state.get("aml_status"),
        "form_fill_seconds": state.get("form_fill_seconds"),
        "keystroke_entropy": state.get("keystroke_entropy"),
        "document_verified": state.get("document_verified", False),
    }

    phone = state.get("phone")
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
        raw = _ollama_generate(prompt)

        # gemma2:2b sometimes wraps output in ```json ... ``` — strip it
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

    except urllib.error.URLError as exc:
        return _fallback(f"Ollama unreachable: {exc}")
    except json.JSONDecodeError as exc:
        return _fallback(f"JSON parse error: {exc}")
    except Exception as exc:
        return _fallback(str(exc))


# ---------------------------------------------------------------------------
# Main check node
# ---------------------------------------------------------------------------

def check_node(state: OnboardingState) -> dict[str, Any]:
    # ── AML gate (unchanged logic) ──────────────────────────────────────────
    aml_status = state.get("aml_status")
    aml_in_background = bool(state.get("aml_in_background"))

    if aml_in_background or aml_status in (None, "pending", "checking"):
        return {
            "stage": "fraud_check",
            "progress": max(state.get("progress", 0), 80),
            "messages": list(state.get("messages", [])) + [
                {
                    "role": "assistant",
                    "text": (
                        "AML screening is in progress. We will continue to final activation "
                        "as soon as compliance checks finish."
                    ),
                }
            ],
        }

    if aml_status == "flagged":
        return {
            "stage": "rejected",
            "fraud_status": "flagged",
            "messages": list(state.get("messages", [])) + [
                {
                    "role": "assistant",
                    "text": "We are unable to proceed with this application after compliance review.",
                }
            ],
        }

    # If fraud already cleared in a prior pass, finalize activation now.
    if state.get("fraud_status") == "clear" and state.get("stage") == "fraud_check":
        return {
            "fraud_status": "clear",
            "fraud_risk_score": state.get("fraud_risk_score"),
            "fraud_signals": state.get("fraud_signals", []),
            "stage": "complete",
            "progress": 100,
            "messages": [
                {
                    "role": "assistant",
                    "text": (
                        "Fraud checks are complete. Your account is now activated. "
                        "Welcome to AccuEntry."
                    ),
                }
            ],
        }

    # ── Layered fraud checks ────────────────────────────────────────────────
    t0 = time.monotonic()

    l1_score, l1_signals = layer1_rule_checks(state)
    l2_score, l2_signals = layer2_behavioural(state)
    l3_score, l3_signals = layer3_identity(state)

    combined_rule_score = l1_score + l2_score + l3_score
    all_signals = l1_signals + l2_signals + l3_signals

    llm_result = layer4_llm_reasoning(combined_rule_score, all_signals, state)
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

    # ── Route on LLM recommendation ─────────────────────────────────────────
    if action == "reject":
        return {
            "stage": "rejected",
            "fraud_status": "rejected",
            "fraud_risk_score": risk_score,
            "fraud_signals": all_signals,
            "messages": list(state.get("messages", [])) + [
                {
                    "role": "assistant",
                    "text": "We are unable to proceed with this application at this time.",
                }
            ],
        }

    if action == "manual_review":
        return {
            "stage": "manual_review",
            "fraud_status": "review",
            "fraud_risk_score": risk_score,
            "fraud_signals": all_signals,
            "fraud_reasoning": reasoning,
            "progress": 92,
            "messages": list(state.get("messages", [])) + [
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
    return {
        "fraud_status": "clear",
        "fraud_risk_score": risk_score,
        "fraud_signals": all_signals,
        "stage": "complete",  # ← FIX: transition to complete, not back to fraud_check
        "progress": 100,
        "messages": list(state.get("messages", [])) + [
            {
                "role": "assistant",
                "text": (
                    "Fraud checks cleared. Your account is now activated. Welcome to AccuEntry."
                ),
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