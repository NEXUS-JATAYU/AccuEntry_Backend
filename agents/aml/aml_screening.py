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
from agents.aml.aml_scoring import compute_risk_score, route_by_score

_memory = AgentMemoryManager()


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
    Fires checks more efficiently:
    1. Run RBI, OFAC, PEP checks in parallel
    2. Evaluate risk rules once with real scores (no placeholder/re-run)
    """
    loop = asyncio.get_event_loop()

    pan_number = state.get("pan_number") or ""
    full_name = state.get("full_name") or ""

    rbi_task = loop.run_in_executor(None, check_rbi_caution_list, pan_number)
    ofac_task = loop.run_in_executor(None, check_ofac_sanctions, full_name)
    pep_task = loop.run_in_executor(None, check_pep_list, full_name)

    # Wait for all three checks to complete
    rbi, ofac, pep = await asyncio.gather(
        rbi_task, ofac_task, pep_task
    )

    # Now run rules once with real scores from OFAC/PEP
    rules = evaluate_risk_rules(
        state.get("full_name") or "",
        state.get("dob"),
        state.get("account_type", ""),
        ofac_match_score=ofac.get("match_score", 0),
        pep_match_score=pep.get("match_score", 0),
    )

    return {
        "aml_status": "checking",
        "aml_completed": False,
        "aml_risk_score": state.get("aml_risk_score") or 0,
        "aml_raw_results": {
            "rbi":   rbi,
            "ofac":  ofac,
            "pep":   pep,
            "rules": rules,
        }
    }


# ── Node 2: aggregate into a score ───────────────────────────────

def aggregate_node(state: OnboardingState) -> dict[str, Any]:
    score = compute_risk_score(state["aml_raw_results"])
    return {
        "aml_status": "checking",
        "aml_completed": False,
        "aml_risk_score": score,
    }


# ── Routing function ─────────────────────────────────────────────

def route_node(state: OnboardingState) -> str:
    return route_by_score(state["aml_risk_score"])


# ── Node 3a: auto clear ───────────────────────────────────────────

def auto_clear_node(state: OnboardingState) -> dict[str, Any]:
    _store_aml_memory(state, final_status="clear", outcome_stage="fraud_check")
    return {
        "aml_status": "clear",
        "aml_risk_score": int(state.get("aml_risk_score") or 0),
        "stage": "fraud_check",
        "progress": 80,
        "messages": [{
            "role": "assistant",
            "text": "AML screening complete — all checks passed. "
                    "Proceeding to the final fraud check."
        }]
    }


# ── Node 3b: auto flag ────────────────────────────────────────────

def auto_flag_node(state: OnboardingState) -> dict[str, Any]:
    _store_aml_memory(state, final_status="flagged", outcome_stage="rejected")
    return {
        "aml_status": "flagged",
        "aml_risk_score": int(state.get("aml_risk_score") or 0),
        "stage": "rejected",
        "progress": 100,
        "messages": [{
            "role": "assistant",
            "text": "Your account is flagged for non compliance. A ticket has been raised. Bank staff will contact you in 1-2 days."
        }]
    }


# ── Node 3c: LLM review for ambiguous cases (score 30–69) ────────

def llm_review_node(state: OnboardingState) -> dict[str, Any]:
    score = state["aml_risk_score"]
    # Deterministic fallback for review band to avoid any LLM-induced stalls.
    if score < 50:
        return auto_clear_node(state)
    return auto_flag_node(state)


# ── Build the subgraph ────────────────────────────────────────────

def build_aml_graph():
    g = StateGraph(OnboardingState)

    g.add_node("run_checks",  run_checks_node)
    g.add_node("aggregate",   aggregate_node)
    g.add_node("auto_clear",  auto_clear_node)
    g.add_node("auto_flag",   auto_flag_node)
    g.add_node("llm_review",  llm_review_node)

    g.set_entry_point("run_checks")
    g.add_edge("run_checks", "aggregate")

    g.add_conditional_edges("aggregate", route_node, {
        "auto_clear": "auto_clear",
        "auto_flag":  "auto_flag",
        "llm_review": "llm_review",
    })

    g.add_edge("auto_clear",  END)
    g.add_edge("auto_flag",   END)
    g.add_edge("llm_review",  END)

    return g.compile()