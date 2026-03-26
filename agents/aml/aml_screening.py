import asyncio
from typing import Any
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from state import OnboardingState
from agents.aml.tools import (
    check_rbi_caution_list,
    check_ofac_sanctions,
    check_pep_list,
    evaluate_risk_rules,
)
from agents.aml.aml_scoring import compute_risk_score, route_by_score

llm = ChatOllama(model="gemma2:2b", max_tokens=256)


# ── Node 1: run all checks concurrently ──────────────────────────

async def run_checks_node(state: OnboardingState) -> dict[str, Any]:
    """
    Fires checks more efficiently:
    1. Run RBI, OFAC, PEP checks in parallel
    2. Evaluate risk rules once with real scores (no placeholder/re-run)
    """
    loop = asyncio.get_event_loop()

    rbi_task  = loop.run_in_executor(None, check_rbi_caution_list,
                                     state["pan_number"])
    ofac_task = loop.run_in_executor(None, check_ofac_sanctions,
                                     state["full_name"])
    pep_task  = loop.run_in_executor(None, check_pep_list,
                                     state["full_name"])

    # Wait for all three checks to complete
    rbi, ofac, pep = await asyncio.gather(
        rbi_task, ofac_task, pep_task
    )

    # Now run rules once with real scores from OFAC/PEP
    rules = evaluate_risk_rules(
        state["full_name"],
        state.get("dob"),
        state.get("account_type", ""),
        ofac_match_score=ofac.get("match_score", 0),
        pep_match_score=pep.get("match_score", 0),
    )

    return {
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
    return {"aml_risk_score": score}


# ── Routing function ─────────────────────────────────────────────

def route_node(state: OnboardingState) -> str:
    return route_by_score(state["aml_risk_score"])


# ── Node 3a: auto clear ───────────────────────────────────────────

def auto_clear_node(state: OnboardingState) -> dict[str, Any]:
    return {
        "aml_status": "clear",
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
    return {
        "aml_status": "flagged",
        "stage": "rejected",
        "progress": state.get("progress", 70),
        "messages": [{
            "role": "assistant",
            "text": "We are unable to proceed with your application at "
                    "this time. Our compliance team will be in touch if "
                    "further action is required."
        }]
    }


# ── Node 3c: LLM review for ambiguous cases (score 30–69) ────────

def llm_review_node(state: OnboardingState) -> dict[str, Any]:
    r = state["aml_raw_results"]
    score = state["aml_risk_score"]

    # Build a structured summary for the LLM — no raw JSON, just clear facts
    signals = []
    if r["rbi"]["hit"]:
        signals.append(f"- RBI caution list: HIT ({r['rbi']['reason']})")
    if r["ofac"]["hit"]:
        signals.append(f"- OFAC sanctions: HIT (score {r['ofac']['match_score']}, "
                       f"program {r['ofac']['program']})")
    elif r["ofac"]["near_miss"]:
        signals.append(f"- OFAC sanctions: near-miss (score {r['ofac']['match_score']})")
    if r["pep"]["hit"]:
        signals.append(f"- PEP: YES — {r['pep']['position']} (tier {r['pep']['pep_tier']})")
    elif r["pep"]["near_miss"]:
        signals.append(f"- PEP: near-miss (score {r['pep']['match_score']})")
    for rule in r["rules"]["triggered_rules"]:
        signals.append(f"- Rule triggered: {rule['rule_id']} — {rule['description']}")

    signals_text = "\n".join(signals) if signals else "- No specific hits, risk from rules only"

    prompt = f"""You are a bank compliance officer reviewing an account application 
that has been flagged for human-equivalent review (risk score {score}/100).

Applicant: {state['full_name']}
DOB: {state.get('dob', 'not provided')}
Account type: {state.get('account_type', 'not provided')}

AML screening signals:
{signals_text}

A score of {score}/100 puts this in the REVIEW band (30–69).
Consider whether the combination of signals warrants rejection or if 
the application can proceed with enhanced due diligence.

Respond with exactly:
Line 1: CLEAR or FLAG
Line 2: One sentence reason (max 20 words)"""

    response = llm.invoke(prompt)
    first_line = response.content.strip().split("\n")[0].strip().upper()

    if "CLEAR" in first_line:
        return auto_clear_node(state)
    else:
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