# agents/aml/tools.py
from datetime import date
from typing import Optional
from rapidfuzz import fuzz
from core.mongodbase import aml_db
import time
from pymongo.errors import OperationFailure

# Risk rules cache (TTL: 1 hour)
_risk_rules_cache = {"data": None, "timestamp": 0, "ttl": 3600}


def _get_cached_risk_rules():
    """Returns cached risk rules or fetches fresh ones if cache expired."""
    now = time.time()
    if (_risk_rules_cache["data"] is None or 
        now - _risk_rules_cache["timestamp"] > _risk_rules_cache["ttl"]):
        _risk_rules_cache["data"] = list(aml_db.risk_rules.find({"active": True}))
        _risk_rules_cache["timestamp"] = now
    return _risk_rules_cache["data"]


def _fallback_search(collection, full_name: str, limit: int = 5):
    """
    Fallback search when text index is missing.
    Returns all active documents (index required setup).
    """
    try:
        return list(collection.find(
            {"active": True},
            {"name": 1, "aliases": 1, "program": 1, "uid": 1, 
             "dob": 1, "position": 1, "pep_tier": 1, "jurisdiction": 1}
        ).limit(limit * 3))  # Get more docs to compensate for lack of ranking
    except Exception:
        return []


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
    Two-stage OFAC check with error handling:
    1. Try MongoDB text index search (fast, requires index)
    2. Fallback to simple search if index missing
    3. rapidfuzz token_sort_ratio with early exit at 85+
    Threshold 85+ = hard hit. 70–84 = near-miss.
    """
    try:
        candidates = list(
            aml_db.ofac_sdn_list.find(
                {"$text": {"$search": full_name}, "active": True},
                {"score": {"$meta": "textScore"}, "name": 1,
                 "aliases": 1, "program": 1, "uid": 1}
            ).sort([("score", {"$meta": "textScore"})]).limit(5)
        )
    except OperationFailure as e:
        if "text index required" in str(e):
            # Fallback: no text index yet, search all active records
            candidates = _fallback_search(aml_db.ofac_sdn_list, full_name)
        else:
            raise

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
            # Early exit if we hit a hard match
            if best_score >= 85:
                break

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
    Fuzzy name match against PEP list with error handling.
    Returns tier (1/2/3) and position for LLM context.
    Threshold 80+ = PEP match. Early exit at 80+.
    """
    try:
        candidates = list(
            aml_db.pep_list.find(
                {"$text": {"$search": full_name}, "active": True},
                {"name": 1, "dob": 1, "position": 1,
                 "pep_tier": 1, "related_to": 1, "jurisdiction": 1}
            ).limit(5)
        )
    except OperationFailure as e:
        if "text index required" in str(e):
            # Fallback: no text index yet, search all active records
            candidates = _fallback_search(aml_db.pep_list, full_name)
        else:
            raise

    best_score = 0
    best_match = None
    query = full_name.lower().strip()

    for candidate in candidates:
        score = fuzz.token_sort_ratio(query, candidate["name"].lower())
        if score > best_score:
            best_score = score
            best_match = candidate
            # Early exit if we hit a PEP match
            if best_score >= 80:
                break

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
    Evaluates configurable risk rules (cached, reloaded every hour).
    """
    rules = _get_cached_risk_rules()  # Use cached rules instead of live query
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