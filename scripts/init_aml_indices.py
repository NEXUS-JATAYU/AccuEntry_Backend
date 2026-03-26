"""
Initialize MongoDB indices for AML screening.
Run this once to set up all required indices.
"""
from core.mongodbase import aml_db

def init_aml_indices():
    """Create all required indices for AML screening."""
    
    print("Initializing AML MongoDB indices...")
    
    # OFAC SDN List indices
    try:
        aml_db.ofac_sdn_list.create_index([("name", "text"), ("aliases", "text")], name="ofac_text_search")
        print("✓ Created text index on ofac_sdn_list (name, aliases)")
    except Exception as e:
        print(f"⚠ ofac_sdn_list text index: {e}")
    
    try:
        aml_db.ofac_sdn_list.create_index([("active", 1), ("uid", 1)], name="ofac_active_uid")
        print("✓ Created compound index on ofac_sdn_list (active, uid)")
    except Exception as e:
        print(f"⚠ ofac_sdn_list compound index: {e}")
    
    # PEP List indices
    try:
        aml_db.pep_list.create_index([("name", "text")], name="pep_text_search")
        print("✓ Created text index on pep_list (name)")
    except Exception as e:
        print(f"⚠ pep_list text index: {e}")
    
    try:
        aml_db.pep_list.create_index([("active", 1), ("pep_tier", 1)], name="pep_active_tier")
        print("✓ Created compound index on pep_list (active, pep_tier)")
    except Exception as e:
        print(f"⚠ pep_list compound index: {e}")
    
    # RBI Caution List indices
    try:
        aml_db.rbi_caution_list.create_index([("pan", 1), ("active", 1)], name="rbi_pan_active")
        print("✓ Created compound index on rbi_caution_list (pan, active)")
    except Exception as e:
        print(f"⚠ rbi_caution_list index: {e}")
    
    # Risk Rules indices
    try:
        aml_db.risk_rules.create_index([("active", 1)], name="risk_rules_active")
        print("✓ Created index on risk_rules (active)")
    except Exception as e:
        print(f"⚠ risk_rules index: {e}")
    
    print("\n✓ AML indices initialization complete!")


if __name__ == "__main__":
    init_aml_indices()
