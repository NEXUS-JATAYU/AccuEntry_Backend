# agents/aml/tools.py
from datetime import date
from typing import Optional
from rapidfuzz import fuzz
from core.mongodbase import aml_db
from agents.aml.aml_scoring import OFAC_FUZZY_HIT_THRESHOLD, PEP_FUZZY_HIT_THRESHOLD
import time
from pymongo.errors import ExecutionTimeout, OperationFailure
import re

# Query/perf tuning constants.
QUERY_MAX_MS = 1200
TEXT_CANDIDATE_LIMIT = 3
_PAN_WS_RE = re.compile(r"\s+")
_NAME_WS_RE = re.compile(r"\s+")

# Risk rules cache (TTL: 5 min)
_risk_rules_cache = {"data": None, "timestamp": 0, "ttl": 300}


def _get_cached_risk_rules():
    """Returns cached risk rules or fetches fresh ones if cache expired."""
    now = time.time()
    if (_risk_rules_cache["data"] is None or 
        now - _risk_rules_cache["timestamp"] > _risk_rules_cache["ttl"]):
        _risk_rules_cache["data"] = list(aml_db.risk_rules.find({"active": True}))
        _risk_rules_cache["timestamp"] = now
    return _risk_rules_cache["data"]


def _fallback_search(collection, full_name: str, limit: int = 5, *, include_aliases: bool = False):
    """
    Fallback search when text index is missing.
    Filters active records by tokenized name prefix (no unfiltered scans).
    """
    query_norm = _normalize_name(full_name)
    if not query_norm:
        return []

    tokens = [re.escape(t) for t in query_norm.split() if len(t) >= 2]
    if not tokens:
        return []

    pattern = ".*".join(tokens)
    or_clauses: list[dict] = [{"name": {"$regex": pattern, "$options": "i"}}]
    if include_aliases:
        or_clauses.append({"aliases": {"$elemMatch": {"$regex": pattern, "$options": "i"}}})

    try:
        cursor = (
            collection.find(
                {"active": True, "$or": or_clauses},
                {
                    "name": 1,
                    "aliases": 1,
                    "program": 1,
                    "uid": 1,
                    "dob": 1,
                    "position": 1,
                    "pep_tier": 1,
                    "jurisdiction": 1,
                },
            )
            .max_time_ms(QUERY_MAX_MS)
            .limit(limit)
        )
        return list(cursor)
    except Exception:
        return []


def _normalize_name(name: str) -> str:
    return _NAME_WS_RE.sub(" ", str(name or "")).strip().lower()


def check_rbi_caution_list(pan: str) -> dict:
    """Exact PAN lookup against RBI caution list."""
    normalized_pan = _PAN_WS_RE.sub("", str(pan or "")).upper().strip()
    if not normalized_pan:
        return {
            "source": "rbi_caution_list",
            "hit": False,
            "reason": None,
            "matched_name": None,
            "bank": None,
        }

    hit = aml_db.rbi_caution_list.find_one(
        {"pan": normalized_pan, "active": True},
        {"_id": 0, "pan": 1, "name": 1, "reason": 1, "bank": 1},
        max_time_ms=QUERY_MAX_MS,
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
    OFAC check with exact-name hit semantics:
    1. Try exact case-insensitive name/alias match (decisioning hit)
    2. Use text/fuzzy search only for near-miss telemetry
    """
    normalized_name = " ".join(str(full_name or "").split()).strip()
    if not normalized_name:
        return {
            "source": "ofac_sdn",
            "hit": False,
            "near_miss": False,
            "match_score": 0,
            "matched_name": None,
            "program": None,
            "uid": None,
        }

    exact_regex = f"^{re.escape(normalized_name)}$"
    try:
        exact_hit = aml_db.ofac_sdn_list.find_one(
            {
                "active": True,
                "$or": [
                    {"name": {"$regex": exact_regex, "$options": "i"}},
                    {"aliases": {"$elemMatch": {"$regex": exact_regex, "$options": "i"}}},
                ],
            },
            {"_id": 0, "name": 1, "program": 1, "uid": 1},
            max_time_ms=QUERY_MAX_MS,
        )
    except ExecutionTimeout:
        exact_hit = None

    if exact_hit:
        return {
            "source": "ofac_sdn",
            "hit": True,
            "near_miss": False,
            "match_score": 100,
            "matched_name": exact_hit.get("name"),
            "program": exact_hit.get("program"),
            "uid": exact_hit.get("uid"),
        }

    try:
        cursor = (
            aml_db.ofac_sdn_list.find(
                {"$text": {"$search": normalized_name}, "active": True},
                {
                    "score": {"$meta": "textScore"},
                    "name": 1,
                    "aliases": 1,
                    "program": 1,
                    "uid": 1,
                },
            )
            .sort([("score", {"$meta": "textScore"})])
            .max_time_ms(QUERY_MAX_MS)
            .limit(TEXT_CANDIDATE_LIMIT)
        )
        candidates = list(cursor)
    except OperationFailure as e:
        if "text index required" in str(e):
            # Fallback: no text index yet, search all active records
            candidates = _fallback_search(
                aml_db.ofac_sdn_list,
                normalized_name,
                limit=TEXT_CANDIDATE_LIMIT,
                include_aliases=True,
            )
        else:
            raise
    except ExecutionTimeout:
        candidates = []

    best_score = 0
    best_match = None
    query = _normalize_name(normalized_name)

    for candidate in candidates:
        # Check primary name
        score = fuzz.token_sort_ratio(query, _normalize_name(candidate.get("name")))
        # Check all aliases too
        for alias in candidate.get("aliases", []):
            alias_score = fuzz.token_sort_ratio(query, _normalize_name(alias))
            score = max(score, alias_score)

        if score > best_score:
            best_score = score
            best_match = candidate
            # Early exit if we hit a hard match
            if best_score >= OFAC_FUZZY_HIT_THRESHOLD:
                break

    fuzzy_hit = best_score >= OFAC_FUZZY_HIT_THRESHOLD
    return {
        "source": "ofac_sdn",
        "hit": fuzzy_hit,
        "near_miss": (not fuzzy_hit) and 70 <= best_score < OFAC_FUZZY_HIT_THRESHOLD,
        "match_score": best_score,
        "matched_name": best_match["name"] if best_match else None,
        "program": best_match.get("program") if best_match else None,
        "uid": best_match.get("uid") if best_match else None,
    }


def check_pep_list(full_name: str) -> dict:
    """
    PEP check with exact-name hit semantics:
    1. Try exact case-insensitive name match (decisioning hit)
    2. Use text/fuzzy search only for near-miss telemetry
    """
    normalized_name = " ".join(str(full_name or "").split()).strip()
    if not normalized_name:
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

    exact_regex = f"^{re.escape(normalized_name)}$"
    try:
        exact_hit = aml_db.pep_list.find_one(
            {
                "active": True,
                "name": {"$regex": exact_regex, "$options": "i"},
            },
            {
                "_id": 0,
                "name": 1,
                "pep_tier": 1,
                "position": 1,
                "jurisdiction": 1,
            },
            max_time_ms=QUERY_MAX_MS,
        )
    except ExecutionTimeout:
        exact_hit = None

    if exact_hit:
        return {
            "source": "pep_list",
            "hit": True,
            "near_miss": False,
            "match_score": 100,
            "pep_tier": exact_hit.get("pep_tier"),
            "position": exact_hit.get("position"),
            "matched_name": exact_hit.get("name"),
            "jurisdiction": exact_hit.get("jurisdiction"),
        }

    try:
        cursor = (
            aml_db.pep_list.find(
                {"$text": {"$search": normalized_name}, "active": True},
                {
                    "name": 1,
                    "dob": 1,
                    "position": 1,
                    "pep_tier": 1,
                    "related_to": 1,
                    "jurisdiction": 1,
                },
            )
            .max_time_ms(QUERY_MAX_MS)
            .limit(TEXT_CANDIDATE_LIMIT)
        )
        candidates = list(cursor)
    except OperationFailure as e:
        if "text index required" in str(e):
            # Fallback: no text index yet, search all active records
            candidates = _fallback_search(aml_db.pep_list, normalized_name, limit=TEXT_CANDIDATE_LIMIT)
        else:
            raise
    except ExecutionTimeout:
        candidates = []

    best_score = 0
    best_match = None
    query = _normalize_name(normalized_name)

    for candidate in candidates:
        score = fuzz.token_sort_ratio(query, _normalize_name(candidate.get("name")))
        if score > best_score:
            best_score = score
            best_match = candidate
            # Early exit if we hit a PEP match
            if best_score >= PEP_FUZZY_HIT_THRESHOLD:
                break

    fuzzy_hit = best_score >= PEP_FUZZY_HIT_THRESHOLD
    return {
        "source": "pep_list",
        "hit": fuzzy_hit,
        "near_miss": (not fuzzy_hit) and 70 <= best_score < PEP_FUZZY_HIT_THRESHOLD,
        "match_score": best_score,
        "pep_tier": best_match.get("pep_tier") if fuzzy_hit and best_match else None,
        "position": best_match.get("position") if fuzzy_hit and best_match else None,
        "matched_name": best_match["name"] if best_match else None,
        "jurisdiction": best_match.get("jurisdiction") if fuzzy_hit and best_match else None,
    }


def evaluate_risk_rules(
    full_name: str,
    dob: Optional[str],
    account_type: str,
    ofac_match_score: int,
    pep_match_score: int,
    politically_exposed: Optional[str] = None,
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

        elif check == "self_declared_pep":
            declared = (politically_exposed or "").strip().lower()
            matched = declared in {"yes", "related to one"}

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