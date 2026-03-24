# seed_mock_db.py
import sys
from pathlib import Path

# Allow running this script from agents/aml or from backend root.
BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.mongodbase import aml_db

# ── helpers ───────────────────────────────────────────────────────
def reset(collection_name):
    aml_db[collection_name].drop()
    print(f"Dropped {collection_name}")

# ─────────────────────────────────────────────────────────────────
# RBI CAUTION LIST
# Real schema: PAN, name, bank account, reason code
# ─────────────────────────────────────────────────────────────────
reset("rbi_caution_list")
aml_db.rbi_caution_list.insert_many([
    # Use these PANs in your dev test cases
    {
        "pan": "FRAUD1234F",
        "name": "Ramesh Kumar Dubey",
        "account_number": "9876543210",
        "bank": "State Bank of India",
        "reason": "wilful_default",
        "amount_crore": 12.5,
        "listed_date": "2023-04-12",
        "active": True
    },
    {
        "pan": "DEFLT5678D",
        "name": "Sunita Mehta Trading Co",
        "account_number": "1122334455",
        "bank": "Punjab National Bank",
        "reason": "fraud",
        "amount_crore": 4.2,
        "listed_date": "2022-08-30",
        "active": True
    },
    {
        "pan": "PONZI9999Z",
        "name": "Vikram Ponzi Schemes",
        "account_number": "9988776655",
        "bank": "HDFC Bank",
        "reason": "fraud",
        "amount_crore": 87.0,
        "listed_date": "2021-11-01",
        "active": True
    },
    # Inactive entry — should NOT trigger
    {
        "pan": "OLDFL0000O",
        "name": "Old Resolved Case",
        "account_number": "0000000000",
        "bank": "Axis Bank",
        "reason": "wilful_default",
        "amount_crore": 1.1,
        "listed_date": "2019-01-01",
        "active": False   # resolved — must not flag
    },
])
aml_db.rbi_caution_list.create_index("pan", unique=False)
aml_db.rbi_caution_list.create_index("account_number")
print("Seeded rbi_caution_list")


# ─────────────────────────────────────────────────────────────────
# OFAC SDN LIST
# Real schema mirrors treasury.gov SDN XML export
# Text index enables fuzzy search across name + aliases
# ─────────────────────────────────────────────────────────────────
reset("ofac_sdn_list")
aml_db.ofac_sdn_list.insert_many([
    # Hard hit — exact name match test case
    {
        "uid": "SDN-001",
        "name": "Mohammad Al Rashid",
        "aliases": ["M. Al Rashid", "Abu Rashid", "Mohammad Rasheed"],
        "dob": "1971-05-14",
        "nationality": "IR",
        "program": "IRAN",
        "entity_type": "individual",
        "active": True
    },
    # Fuzzy hit test case — applicant uses slightly different spelling
    {
        "uid": "SDN-002",
        "name": "Dawood Ibrahim Kaskar",
        "aliases": ["D. Ibrahim", "Dawood Ebrahim"],
        "dob": "1955-12-26",
        "nationality": "PK",
        "program": "SDGT",
        "entity_type": "individual",
        "active": True
    },
    # Corporate entity
    {
        "uid": "SDN-003",
        "name": "Golden Shield Import Export LLC",
        "aliases": ["Golden Shield LLC", "GS Import Export"],
        "dob": None,
        "nationality": "AE",
        "program": "SYRIA",
        "entity_type": "entity",
        "active": True
    },
    # Near-miss — similar name but should NOT cross 85 threshold
    {
        "uid": "SDN-004",
        "name": "Raj Kumar Singh Sanction",
        "aliases": [],
        "dob": "1980-03-01",
        "nationality": "IN",
        "program": "TEST",
        "entity_type": "individual",
        "active": True
    },
])
# Text index across name AND aliases array — critical for fuzzy search
aml_db.ofac_sdn_list.create_index(
    [("name", "text"), ("aliases", "text")],
    weights={"name": 10, "aliases": 5}
)
print("Seeded ofac_sdn_list")


# ─────────────────────────────────────────────────────────────────
# PEP LIST
# Tier 1 = direct PEP (politician, judge, senior official)
# Tier 2 = immediate family of tier 1
# Tier 3 = known associate of tier 1
# ─────────────────────────────────────────────────────────────────
reset("pep_list")
aml_db.pep_list.insert_many([
    # Tier 1 — sitting MP
    {
        "name": "Arun Kumar Singh",
        "dob": "1965-03-22",
        "position": "Member of Parliament",
        "jurisdiction": "Lok Sabha",
        "country": "IN",
        "pep_tier": 1,
        "active": True
    },
    # Tier 1 — IAS Secretary level
    {
        "name": "Priya Sharma",
        "dob": "1972-07-15",
        "position": "Secretary, Ministry of Finance",
        "jurisdiction": "Central Government",
        "country": "IN",
        "pep_tier": 1,
        "active": True
    },
    # Tier 1 — State Chief Minister
    {
        "name": "Rajendra Prasad Yadav",
        "dob": "1958-11-04",
        "position": "Chief Minister",
        "jurisdiction": "Bihar",
        "country": "IN",
        "pep_tier": 1,
        "active": True
    },
    # Tier 2 — spouse of tier 1
    {
        "name": "Sunita Arun Singh",
        "dob": "1968-09-12",
        "position": "Spouse of MP Arun Kumar Singh",
        "jurisdiction": "N/A",
        "country": "IN",
        "pep_tier": 2,
        "related_to": "Arun Kumar Singh",
        "active": True
    },
    # Tier 2 — son of CM
    {
        "name": "Rahul Rajendra Yadav",
        "dob": "1988-02-20",
        "position": "Son of CM Rajendra Prasad Yadav",
        "jurisdiction": "N/A",
        "country": "IN",
        "pep_tier": 2,
        "related_to": "Rajendra Prasad Yadav",
        "active": True
    },
    # Tier 3 — business associate
    {
        "name": "Deepak Malhotra",
        "dob": "1975-06-30",
        "position": "Known associate, contracts with state govt",
        "jurisdiction": "Bihar",
        "country": "IN",
        "pep_tier": 3,
        "related_to": "Rajendra Prasad Yadav",
        "active": True
    },
    # Former PEP — still active in list but no longer in office
    {
        "name": "Anjali Verma",
        "dob": "1960-01-18",
        "position": "Former High Court Judge (retired 2022)",
        "jurisdiction": "Allahabad High Court",
        "country": "IN",
        "pep_tier": 1,
        "active": True   # former PEPs remain listed
    },
])
aml_db.pep_list.create_index([("name", "text")])
aml_db.pep_list.create_index("pep_tier")
print("Seeded pep_list")


# ─────────────────────────────────────────────────────────────────
# RISK RULES
# Configurable — tune score_delta without touching agent code
# ─────────────────────────────────────────────────────────────────
reset("risk_rules")
aml_db.risk_rules.insert_many([
    # Age/account type mismatch rules
    {
        "rule_id": "MINOR_APPLICANT",
        "description": "Applicant is under 18",
        "check_type": "age_threshold",
        "params": {"max_age": 17, "operator": "lte"},
        "risk_score_delta": 60,
        "active": True
    },
    {
        "rule_id": "BUSINESS_ACCOUNT_YOUNG",
        "description": "Business account for applicant under 21",
        "check_type": "age_account_combo",
        "params": {"max_age": 20, "account_types": ["business", "current"]},
        "risk_score_delta": 25,
        "active": True
    },
    {
        "rule_id": "SENIOR_HIGH_RISK_PRODUCT",
        "description": "Applicant over 75 applying for credit card",
        "check_type": "age_account_combo",
        "params": {"min_age": 75, "account_types": ["credit_card"]},
        "risk_score_delta": 15,
        "active": True
    },
    # Name fuzzy match ambiguity rule
    # (applied when sanctions match score is 70–84 — below hard threshold)
    {
        "rule_id": "NEAR_SANCTIONS_MATCH",
        "description": "Name is similar to a sanctions list entry",
        "check_type": "fuzzy_score_range",
        "params": {"min_score": 70, "max_score": 84, "source": "ofac_sdn"},
        "risk_score_delta": 30,
        "active": True
    },
    {
        "rule_id": "NEAR_PEP_MATCH",
        "description": "Name is similar to a PEP entry",
        "check_type": "fuzzy_score_range",
        "params": {"min_score": 70, "max_score": 79, "source": "pep"},
        "risk_score_delta": 15,
        "active": True
    },
])
aml_db.risk_rules.create_index("rule_id", unique=True)
aml_db.risk_rules.create_index("active")
print("Seeded risk_rules")

print("\nAll collections seeded. Test PAN numbers:")
print("  CLEANPAN1A  → not in any list, clean pass")
print("  FRAUD1234F  → RBI caution list hit, auto-flag")
print("  DEFLT5678D  → RBI caution list hit, auto-flag")
print("  Use 'Arun Kumar Singh' as name → PEP tier 1 hit, LLM review")
print("  Use 'Mohammad Al Rashid' as name → OFAC hit, auto-flag")