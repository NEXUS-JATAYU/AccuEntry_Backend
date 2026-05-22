import asyncio
from typing import Any
from langgraph.graph import StateGraph, END
from state import OnboardingState
from memory_manager import AgentMemoryManager
from agents.aml.tools import (
    check_rbi_caution_list,
    check_ofac_sanctions,
    check_pep_list,
    evaluate_risk_rules,
)
from agents.aml.aml_scoring import compute_risk_score, route_after_aggregate, has_screening_failure
from rag_service import retrieve_as_context

_memory = AgentMemoryManager()

_RBI_TIMEOUT_S = 2.0
_OFAC_PEP_TIMEOUT_S = 3.5


def _empty_ofac() -> dict:
    return {
        "source": "ofac_sdn",
        "hit": False,
        "near_miss": False,
        "match_score": 0,
        "matched_name": None,
        "program": None,
        "uid": None,
    }


def _empty_pep() -> dict:
    return {
        "source": "pep_list",
        "hit": False,
        "near_miss": False,
        "match_score": 0,
        "pep_tier": None,
        "position": None,
        "matched_name": None,
        "jurisdiction": None,
    }


def _empty_rbi() -> dict:
    return {
        "source": "rbi_caution_list",
        "hit": False,
        "reason": None,
        "matched_name": None,
        "bank": None,
    }


def _evaluate_rules(state: OnboardingState, ofac: dict, pep: dict) -> dict:
    return evaluate_risk_rules(
        state.get("full_name") or "",
        state.get("dob"),
        state.get("account_type", ""),
        ofac_match_score=ofac.get("match_score", 0),
        pep_match_score=pep.get("match_score", 0),
        politically_exposed=state.get("politically_exposed"),
    )


def _store_aml_memory(state: OnboardingState, final_status: str, outcome_stage: str) -> None:
    session_id = state.get("audit_session_id") or state.get("session_id") or "unknown"
    aml_results = state.get("aml_raw_results") or {}
    rules = ((aml_results.get("rules") or {}).get("triggered_rules") or [])
    _memory.store_interaction(
        session_id=session_id,
        agent_name="aml_screening",
        input_data={
            "aml_raw_results": aml_results,
            "rule_count": len(rules),
            "aml_in_background": bool(state.get("aml_in_background")),
            "full_name": state.get("full_name"),
            "pan_number": state.get("pan_number"),
            "politically_exposed": state.get("politically_exposed"),
        },
        output_data={
            "aml_status": final_status,
            "aml_risk_score": int(state.get("aml_risk_score") or 0),
            "stage": outcome_stage,
        },
        risk_score=float(state.get("aml_risk_score") or 0),
        decision=final_status,
        metadata={
            "audit_session_id": state.get("audit_session_id") or session_id,
            "workflow_stage": "aml_screening",
            "outcome_stage": outcome_stage,
            "triggered_rule_count": len(rules),
        },
        event_type="aml_outcome",
    )


# ── Node 1: run all checks concurrently ──────────────────────────

async def run_checks_node(state: OnboardingState) -> dict[str, Any]:
    """
    Runs RBI, OFAC, PEP checks and configurable risk rules.
    Timeouts are fail-closed (marked error → manual review in routing).
    """
    loop = asyncio.get_running_loop()

    pan_number = state.get("pan_number") or ""
    full_name = state.get("full_name") or ""

    try:
        rbi = await asyncio.wait_for(
            loop.run_in_executor(None, check_rbi_caution_list, pan_number),
            timeout=_RBI_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        rbi = {**_empty_rbi(), "error": "timeout"}

    if rbi.get("hit"):
        rules = _evaluate_rules(state, _empty_ofac(), _empty_pep())
        return {
            "aml_status": "checking",
            "aml_completed": False,
            "aml_risk_score": state.get("aml_risk_score") or 0,
            "aml_raw_results": {
                "rbi": rbi,
                "ofac": _empty_ofac(),
                "pep": _empty_pep(),
                "rules": rules,
            },
        }

    ofac_task = loop.run_in_executor(None, check_ofac_sanctions, full_name)
    pep_task = loop.run_in_executor(None, check_pep_list, full_name)

    try:
        ofac, pep = await asyncio.wait_for(
            asyncio.gather(ofac_task, pep_task),
            timeout=_OFAC_PEP_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        ofac = {**_empty_ofac(), "error": "timeout"}
        pep = {**_empty_pep(), "error": "timeout"}

    rules = _evaluate_rules(state, ofac, pep)

    return {
        "aml_status": "checking",
        "aml_completed": False,
        "aml_risk_score": state.get("aml_risk_score") or 0,
        "aml_raw_results": {
            "rbi": rbi,
            "ofac": ofac,
            "pep": pep,
            "rules": rules,
        },
    }


# ── Node 2: aggregate into a score ───────────────────────────────

def aggregate_node(state: OnboardingState) -> dict[str, Any]:
    raw = state.get("aml_raw_results") or {}
    score = compute_risk_score(raw)
    return {"aml_risk_score": score}


# ── Routing function ─────────────────────────────────────────────

def route_node(state: OnboardingState) -> str:
    return route_after_aggregate(state)


# ── Node 3a: auto clear ───────────────────────────────────────────

def auto_clear_node(state: OnboardingState) -> dict[str, Any]:
    _store_aml_memory(state, final_status="clear", outcome_stage="fraud_check")
    return {
        "aml_status": "clear",
        "aml_completed": True,
        "aml_risk_score": int(state.get("aml_risk_score") or 0),
        "stage": "fraud_check",
        "progress": 80,
        "messages": [{
            "role": "assistant",
            "text": "AML screening complete — all checks passed. "
                    "Proceeding to the final fraud check."
        }],
    }


# ── Node 3b: auto flag ────────────────────────────────────────────

async def auto_flag_node(state: OnboardingState) -> dict[str, Any]:
    from agents.aml.aml_user_report import build_aml_flag_user_message

    raw_aml = state.get("aml_raw_results") or {}
    pep_hit = (raw_aml.get("pep") or {}).get("hit")
    ofac_hit = (raw_aml.get("ofac") or {}).get("hit")
    rbi_hit = (raw_aml.get("rbi") or {}).get("hit")
    flag_type = (
        "PEP" if pep_hit
        else "OFAC sanctions" if ofac_hit
        else "RBI caution list" if rbi_hit
        else "AML"
    )

    try:
        policy_excerpt = retrieve_as_context(f"AML rules for {flag_type}", top_k=3)
    except Exception as e:
        policy_excerpt = "Standard AML non-compliance policy applied."
        print(f"Failed to retrieve policy: {e}")

    report_text = await build_aml_flag_user_message(state)

    _store_aml_memory(state, final_status="flagged", outcome_stage="rejected")
    return {
        "aml_status": "flagged",
        "aml_completed": True,
        "aml_risk_score": int(state.get("aml_risk_score") or 0),
        "aml_policy_excerpt": policy_excerpt,
        "stage": "rejected",
        "progress": 100,
        "decision_action": "reject",
        "decision_reason": f"AML flagged — primary source: {flag_type}",
        "messages": [{"role": "assistant", "text": report_text}],
    }


# ── Node 3c: manual review for ambiguous cases (score 30–69) ────

def manual_review_node(state: OnboardingState) -> dict[str, Any]:
    raw = state.get("aml_raw_results") or {}

    if has_screening_failure(raw):
        reason = "One or more AML list checks could not be completed in time."
    else:
        reason = "Elevated AML risk score requires compliance review."

    _store_aml_memory(state, final_status="review", outcome_stage="manual_review")
    return {
        "aml_status": "review",
        "aml_completed": True,
        "aml_risk_score": int(state.get("aml_risk_score") or 0),
        "stage": "manual_review",
        "progress": 85,
        "messages": [{
            "role": "assistant",
            "text": f"{reason} Your application has been routed for manual compliance review.",
        }],
    }


# ── Build the subgraph ────────────────────────────────────────────

def build_aml_graph():
    g = StateGraph(OnboardingState)

    g.add_node("run_checks", run_checks_node)
    g.add_node("aggregate", aggregate_node)
    g.add_node("auto_clear", auto_clear_node)
    g.add_node("auto_flag", auto_flag_node)
    g.add_node("manual_review", manual_review_node)

    g.set_entry_point("run_checks")
    g.add_edge("run_checks", "aggregate")

    g.add_conditional_edges("aggregate", route_node, {
        "auto_clear": "auto_clear",
        "auto_flag": "auto_flag",
        "manual_review": "manual_review",
    })

    g.add_edge("auto_clear", END)
    g.add_edge("auto_flag", END)
    g.add_edge("manual_review", END)

    return g.compile()
