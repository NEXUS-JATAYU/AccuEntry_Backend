"""
User-facing AML flag report for chat bubbles.

Builds a structured summary from aml_raw_results, optionally reformatted by LLM.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from agents.faq.faq_agent import POST_PROCESS_FAQ_INVITE
from state import OnboardingState

logger = logging.getLogger(__name__)

_AML_REPORT_LLM_PROMPT = """You format AML screening outcomes for bank customers.
Use ONLY the facts in the JSON below. Do not invent checks, names, or reasons.

Output plain text suitable for a chat message with this structure:
1) Short title line (e.g. "AML Screening Report")
2) Reference ID and risk score
3) Section "Where you were flagged" — bullet each failed/important check with check name and reason
4) Section "Why your application cannot proceed online" — 1-3 sentences from the facts
5) Section "What to do next" — MUST state exactly: "You have been assigned to {assigned_employee_name} and at {assigned_bank_branch} at {assigned_date} from {assigned_time}." using the exact fields from the JSON. DO NOT say "visit your nearest branch".
6) End with exactly one line inviting process questions (use the invite text provided in facts if present)

Keep tone professional and clear. No markdown headers with #. Use bullet characters (•) and line breaks. Max 220 words."""


def extract_aml_flag_details(state: OnboardingState) -> dict[str, Any]:
    """Structured hits from aml_raw_results for report generation."""
    raw = state.get("aml_raw_results") or {}
    if not isinstance(raw, dict):
        raw = {}

    checks: list[dict[str, str]] = []
    flagged_checks: list[dict[str, str]] = []

    rbi = raw.get("rbi") or {}
    if rbi:
        status = "failed" if rbi.get("hit") else ("error" if rbi.get("error") else "clear")
        entry = {
            "name": "RBI Caution List (India)",
            "status": status,
            "detail": str(rbi.get("reason") or rbi.get("error") or "No match on RBI caution list"),
        }
        checks.append(entry)
        if status in {"failed", "error"}:
            flagged_checks.append(entry)

    ofac = raw.get("ofac") or {}
    if ofac:
        if ofac.get("error"):
            status = "error"
        elif ofac.get("hit"):
            status = "failed"
        elif ofac.get("near_miss"):
            status = "review"
        else:
            status = "clear"
        detail_parts = []
        if ofac.get("matched_name"):
            detail_parts.append(f"Matched: {ofac['matched_name']}")
        if ofac.get("match_score") is not None:
            detail_parts.append(f"Score: {ofac['match_score']}")
        if ofac.get("error"):
            detail_parts.append(str(ofac["error"]))
        if not detail_parts:
            detail_parts.append("No OFAC sanctions match")
        entry = {
            "name": "OFAC Sanctions Screening",
            "status": status,
            "detail": " — ".join(detail_parts),
        }
        checks.append(entry)
        if status in {"failed", "error", "review"}:
            flagged_checks.append(entry)

    pep = raw.get("pep") or {}
    if pep:
        if pep.get("error"):
            status = "error"
        elif pep.get("hit"):
            status = "failed"
        elif pep.get("near_miss"):
            status = "review"
        else:
            status = "clear"
        detail_parts = []
        if pep.get("matched_name"):
            detail_parts.append(f"Name: {pep['matched_name']}")
        if pep.get("position"):
            detail_parts.append(f"Role: {pep['position']}")
        if pep.get("pep_tier"):
            detail_parts.append(f"PEP tier: {pep['pep_tier']}")
        if pep.get("error"):
            detail_parts.append(str(pep["error"]))
        if not detail_parts:
            detail_parts.append("No PEP match")
        entry = {
            "name": "Politically Exposed Person (PEP) Screening",
            "status": status,
            "detail": " — ".join(detail_parts),
        }
        checks.append(entry)
        if status in {"failed", "error", "review"}:
            flagged_checks.append(entry)

    rules_block = raw.get("rules") or {}
    triggered = (
        rules_block.get("triggered_rules")
        or rules_block.get("triggered")
        or []
    )
    rule_lines: list[str] = []
    if isinstance(triggered, list):
        for rule in triggered[:8]:
            if isinstance(rule, dict):
                rid = rule.get("rule_id") or rule.get("id") or "rule"
                desc = rule.get("description") or rule.get("reason") or ""
                rule_lines.append(f"{rid}: {desc}".strip(": "))
            else:
                rule_lines.append(str(rule))

    primary = flagged_checks[0]["name"] if flagged_checks else "AML compliance screening"
    if not flagged_checks and rule_lines:
        primary = "Risk policy rules"

    assigned_name = state.get("assigned_employee_name")
    assigned_branch = state.get("assigned_bank_branch")
    assigned_date = state.get("assigned_date")
    assigned_time = state.get("assigned_time")
    if not assigned_name:
        try:
            from core.database import SessionLocal
            from models.employee import Employee, CaseAssignment
            from sqlalchemy import func
            from datetime import datetime
            with SessionLocal() as db:
                sid = state.get("session_id")
                if sid:
                    emp = db.query(Employee).order_by(func.random()).first()
                    if not emp:
                        emp = Employee(email="compliance@accuentry.com", name="System Manager", task="Review", section="Compliance", bank_branch="Main Branch")
                        db.add(emp)
                        db.commit()
                        db.refresh(emp)
                    existing = db.query(CaseAssignment).filter(CaseAssignment.session_id == sid).first()
                    if not existing:
                        assignment = CaseAssignment(session_id=sid, employee_id=emp.id)
                        db.add(assignment)
                        db.commit()
                    assigned_name = emp.name
                    assigned_branch = emp.bank_branch
                    assigned_date = datetime.now().strftime("%Y-%m-%d")
                    assigned_time = datetime.now().strftime("%I:%M %p")
                    # Update state dict directly so it persists upstream if possible
                    state["assigned_employee_name"] = assigned_name
                    state["assigned_bank_branch"] = assigned_branch
                    state["assigned_date"] = assigned_date
                    state["assigned_time"] = assigned_time
        except Exception as e:
            logger.error(f"Error assigning employee: {e}")

    return {
        "reference_id": (state.get("audit_session_id") or state.get("session_id") or "N/A")[:8].upper(),
        "aml_risk_score": int(state.get("aml_risk_score") or 0),
        "all_checks": checks,
        "flagged_checks": flagged_checks,
        "triggered_rules": rule_lines,
        "primary_flag_source": primary,
        "faq_invite": POST_PROCESS_FAQ_INVITE,
        "assigned_employee_name": assigned_name or "a manager",
        "assigned_bank_branch": assigned_branch or "our branch",
        "assigned_date": assigned_date or "today",
        "assigned_time": assigned_time or "now",
    }


def build_aml_flag_report_deterministic(details: dict[str, Any]) -> str:
    """Formatted report without LLM (always available)."""
    ref = details.get("reference_id", "N/A")
    score = details.get("aml_risk_score", 0)
    flagged = details.get("flagged_checks") or []
    rules = details.get("triggered_rules") or []
    primary = details.get("primary_flag_source", "AML screening")

    lines = [
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "       AML SCREENING REPORT",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"Reference ID: {ref}",
        f"Risk score: {score}/100",
        "",
        "STATUS: Your application cannot be completed online.",
        "",
        "WHERE YOU WERE FLAGGED:",
    ]

    if flagged:
        for item in flagged:
            label = "MATCH" if item.get("status") == "failed" else item.get("status", "alert").upper()
            lines.append(f"  • {item['name']} — {label}")
            lines.append(f"    {item['detail']}")
    else:
        lines.append(f"  • {primary} — elevated risk")

    if rules:
        lines.extend(["", "TRIGGERED POLICY RULES:"])
        for rule in rules:
            lines.append(f"  • {rule}")

    all_checks = details.get("all_checks") or []
    cleared = [c for c in all_checks if c not in flagged]
    if cleared:
        lines.extend(["", "OTHER CHECKS (for your reference):"])
        for item in cleared:
            lines.append(f"  • {item['name']}: {item.get('status', 'clear')}")

    lines.extend([
        "",
        "WHY:",
        f"  Screening found a compliance issue under {primary}. "
        "Regulations require in-person verification before an account can be opened.",
        "",
        "WHAT TO DO NEXT:",
        f"  You have been assigned to {details.get('assigned_employee_name')} and at {details.get('assigned_bank_branch')} at {details.get('assigned_date')} from {details.get('assigned_time')}",
        "",
        details.get("faq_invite", POST_PROCESS_FAQ_INVITE),
    ])
    return "\n".join(lines)


async def build_aml_flag_user_message(
    state: OnboardingState,
    *,
    use_llm: bool | None = None,
) -> str:
    """
    User-facing AML flag report for assistant chat bubbles.
    Falls back to deterministic formatting if LLM is disabled or fails.
    """
    details = extract_aml_flag_details(state)
    deterministic = build_aml_flag_report_deterministic(details)

    if use_llm is None:
        use_llm = os.getenv("AML_FLAG_REPORT_USE_LLM", "true").lower() in {
            "1",
            "true",
            "yes",
        }

    if not use_llm:
        return deterministic

    try:
        llm = ChatOllama(
            model=os.getenv("OLLAMA_MODEL", os.getenv("AGENT_LLM_MODEL", "llama3.2")),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=0.1,
        )
        response = await llm.ainvoke([
            SystemMessage(content=_AML_REPORT_LLM_PROMPT),
            HumanMessage(content=json.dumps(details, indent=2)),
        ])
        text = str(getattr(response, "content", "") or "").strip()
        if text and len(text) > 80:
            if POST_PROCESS_FAQ_INVITE not in text:
                text = f"{text}\n\n{POST_PROCESS_FAQ_INVITE}"
            return text
    except Exception as exc:
        logger.warning("AML flag report LLM formatting failed: %s", exc)

    return deterministic
