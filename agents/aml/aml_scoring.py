
# Score weights — tune these without touching graph logic
RBI_HIT_SCORE       = 80   # near-certain reject
OFAC_HIT_SCORE      = 90   # hard block
PEP_TIER_SCORES     = {1: 40, 2: 20, 3: 10}

# Fuzzy match thresholds (also used in tools.py for hit semantics)
OFAC_FUZZY_HIT_THRESHOLD = 85
PEP_FUZZY_HIT_THRESHOLD = 80

# Routing thresholds
AUTO_CLEAR_THRESHOLD = 30   # score < 30  → auto clear, no LLM needed
AUTO_FLAG_THRESHOLD  = 70   # score >= 70 → auto flag, no LLM needed
# 30–69 → manual compliance review


def compute_risk_score(raw_results: dict) -> int:
    """
    Pure function — takes raw tool outputs, returns integer 0–100.
    Called by aggregate_score node in the graph.
    """
    if not raw_results or not isinstance(raw_results, dict):
        return 0

    score = 0
    rbi   = raw_results.get("rbi", {})
    ofac  = raw_results.get("ofac", {})
    pep   = raw_results.get("pep", {})
    rules = raw_results.get("rules", {})

    if rbi.get("hit"):
        score += RBI_HIT_SCORE

    if ofac.get("hit"):
        score += OFAC_HIT_SCORE
    # near_miss already handled by risk rules (NEAR_SANCTIONS_MATCH rule)

    if pep.get("hit"):
        tier = pep.get("pep_tier") or 1
        score += PEP_TIER_SCORES.get(tier, 10)
    # near_miss handled by NEAR_PEP_MATCH rule

    score += rules.get("total_delta", 0)

    return min(score, 100)


def has_screening_failure(raw_results: dict) -> bool:
    """True when a list check timed out or failed — routes to manual review."""
    if not raw_results:
        return False
    for key in ("rbi", "ofac", "pep"):
        check = raw_results.get(key) or {}
        if check.get("error") in {"timeout", "query_failed"}:
            return True
    return False


def route_by_score(score: int) -> str:
    if score < AUTO_CLEAR_THRESHOLD:
        return "auto_clear"
    if score >= AUTO_FLAG_THRESHOLD:
        return "auto_flag"
    return "manual_review"


def route_after_aggregate(state: dict) -> str:
    """Route AML graph after scoring; screening failures always go to manual review."""
    raw = state.get("aml_raw_results") or {}
    if has_screening_failure(raw):
        return "manual_review"
    return route_by_score(int(state.get("aml_risk_score") or 0))
