
# Score weights — tune these without touching graph logic
RBI_HIT_SCORE       = 80   # near-certain reject
OFAC_HIT_SCORE      = 90   # hard block
PEP_TIER_SCORES     = {1: 40, 2: 20, 3: 10}

# Routing thresholds
AUTO_CLEAR_THRESHOLD = 30   # score < 30  → auto clear, no LLM needed
AUTO_FLAG_THRESHOLD  = 70   # score >= 70 → auto flag, no LLM needed
# 30–69 → LLM reviews


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


def route_by_score(score: int) -> str:
    if score < AUTO_CLEAR_THRESHOLD:
        return "auto_clear"
    if score >= AUTO_FLAG_THRESHOLD:
        return "auto_flag"
    return "llm_review"