# agents/aml/tools.py
from datetime import date
from typing import Optional
from rapidfuzz import fuzz
from core.mongodbase import aml_db


def check_rbi_caution_list(pan: str) -> dict:
    """Exact PAN lookup against RBI caution list."""
    hit = aml_db.rbi_caution_list.find_one(
        {"pan": pan.upper(), "active": True},
        {"_id": 0, "pan": 1, "name": 1, "reason": 1, "bank": 1}
    )
    return {
        "source": "rbi_caution_list",
        "hit": bool(hit),
        "reason": hit.get("reason") if hit else None,
        "matched_name": hit.get("name") if hit else None,
        "bank": hit.get("bank") if hit else None,
    }


def check_ofac_sanctions(full_name: str) -> dict:
    """
    Two-stage OFAC check:
    1. MongoDB text index search to get candidates
    2. rapidfuzz token_sort_ratio for fuzzy match against name + aliases
    Threshold 85+ = hard hit. 70–84 = near-miss (handled in risk rules).
    """
    candidates = list(
        aml_db.ofac_sdn_list.find(
            {"$text": {"$search": full_name}, "active": True},
            {"score": {"$meta": "textScore"}, "name": 1,
             "aliases": 1, "program": 1, "uid": 1}
        ).sort([("score", {"$meta": "textScore"})]).limit(10)
    )

    best_score = 0
    best_match = None
    query = full_name.lower().strip()

    for candidate in candidates:
        # Check primary name
        score = fuzz.token_sort_ratio(query, candidate["name"].lower())
        # Check all aliases too
        for alias in candidate.get("aliases", []):
            alias_score = fuzz.token_sort_ratio(query, alias.lower())
            score = max(score, alias_score)

        if score > best_score:
            best_score = score
            best_match = candidate

    return {
        "source": "ofac_sdn",
        "hit": best_score >= 85,
        "near_miss": 70 <= best_score < 85,
        "match_score": best_score,
        "matched_name": best_match["name"] if best_match else None,
        "program": best_match.get("program") if best_match else None,
        "uid": best_match.get("uid") if best_match else None,
    }


def check_pep_list(full_name: str) -> dict:
    """
    Fuzzy name match against PEP list.
    Returns tier (1/2/3) and position for LLM context.
    Threshold 80+ = PEP match.
    """
    candidates = list(
        aml_db.pep_list.find(
            {"$text": {"$search": full_name}, "active": True},
            {"name": 1, "dob": 1, "position": 1,
             "pep_tier": 1, "related_to": 1, "jurisdiction": 1}
        ).limit(10)
    )

    best_score = 0
    best_match = None
    query = full_name.lower().strip()

    for candidate in candidates:
        score = fuzz.token_sort_ratio(query, candidate["name"].lower())
        if score > best_score:
            best_score = score
            best_match = candidate

    hit = best_score >= 80
    return {
        "source": "pep_list",
        "hit": hit,
        "near_miss": 70 <= best_score < 80,
        "match_score": best_score,
        "pep_tier": best_match.get("pep_tier") if hit else None,
        "position": best_match.get("position") if hit else None,
        "matched_name": best_match["name"] if hit else None,
        "jurisdiction": best_match.get("jurisdiction") if hit else None,
    }


def evaluate_risk_rules(
    full_name: str,
    dob: Optional[str],
    account_type: str,
    ofac_match_score: int,
    pep_match_score: int,
) -> dict:
    """
    Evaluates configurable risk rules stored in MongoDB.
    Reads rule definitions at runtime so you can tune without redeploying.
    """
    rules = list(aml_db.risk_rules.find({"active": True}))
    triggered = []
    total_delta = 0

    # Compute age once
    age = 0
    if dob:
        try:
            birth = date.fromisoformat(dob)
            age = (date.today() - birth).days // 365
        except ValueError:
            pass

    for rule in rules:
        matched = False
        p = rule.get("params", {})
        check = rule["check_type"]

        if check == "age_threshold":
            if p.get("operator") == "lte":
                matched = age <= p.get("max_age", 0)
            elif p.get("operator") == "gte":
                matched = age >= p.get("min_age", 999)

        elif check == "age_account_combo":
            account_match = account_type.lower() in [
                a.lower() for a in p.get("account_types", [])
            ]
            if "max_age" in p:
                matched = account_match and age <= p["max_age"]
            elif "min_age" in p:
                matched = account_match and age >= p["min_age"]

        elif check == "fuzzy_score_range":
            score = (ofac_match_score if p.get("source") == "ofac_sdn"
                     else pep_match_score)
            matched = p["min_score"] <= score <= p["max_score"]

        if matched:
            triggered.append({
                "rule_id": rule["rule_id"],
                "description": rule["description"],
                "delta": rule["risk_score_delta"]
            })
            total_delta += rule["risk_score_delta"]

    return {
        "source": "risk_rules",
        "triggered_rules": triggered,
        "total_delta": total_delta,
    }