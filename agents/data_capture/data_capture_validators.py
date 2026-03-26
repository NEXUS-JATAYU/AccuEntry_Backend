import re
from datetime import datetime
from typing import Optional, Tuple


def _normalize_spaces(value: str) -> str:
    return " ".join((value or "").strip().split())


def validate_name(value: str) -> Tuple[bool, str]:
    cleaned = _normalize_spaces(value)
    if len(cleaned) < 3:
        return False, "Name is too short."
    if not re.fullmatch(r"[A-Za-z .'-]+", cleaned):
        return False, "Name can contain letters, spaces, apostrophes, dots, and hyphens only."
    return True, cleaned.upper()


def validate_pan(value: str) -> Tuple[bool, str]:
    candidate = _normalize_spaces(value).upper()
    if not re.fullmatch(r"[A-Z]{5}[0-9]{4}[A-Z]", candidate):
        return False, "Invalid PAN format. Use 10 characters like ABCDE1234F."
    return True, candidate


def validate_date(value: str) -> Tuple[bool, str]:
    raw = _normalize_spaces(value)
    try:
        datetime.strptime(raw, "%Y-%m-%d")
    except ValueError:
        return False, "Invalid date format. Use YYYY-MM-DD."
    return True, raw


def validate_yes_no(value: str, label: str = "value") -> Tuple[bool, str]:
    normalized = _normalize_spaces(value).lower()
    mapping = {
        "yes": "Yes",
        "y": "Yes",
        "true": "Yes",
        "no": "No",
        "n": "No",
        "false": "No",
    }
    if normalized not in mapping:
        return False, f"Invalid {label}. Please enter Yes or No."
    return True, mapping[normalized]


def validate_choice(value: str, choices: list[str], label: str) -> Tuple[bool, str]:
    candidate = _normalize_spaces(value)
    canonical = {c.lower(): c for c in choices}
    if candidate.lower() in canonical:
        return True, canonical[candidate.lower()]
    return False, f"Invalid {label}. Choose one of: {', '.join(choices)}."


def validate_mobile_number(value: str) -> Tuple[bool, str]:
    digits = re.sub(r"\D", "", value or "")
    if len(digits) == 12 and digits.startswith("91"):
        digits = digits[2:]
    if len(digits) != 10:
        return False, "Invalid mobile number. Please enter a 10-digit number."
    if digits[0] not in "6789":
        return False, "Invalid mobile number. It should start with 6, 7, 8, or 9."
    return True, digits


def validate_email(value: str) -> Tuple[bool, str]:
    candidate = _normalize_spaces(value).lower()
    if not re.fullmatch(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", candidate):
        return False, "Invalid email address format."
    return True, candidate


def validate_amount(value: str, label: str = "amount") -> Tuple[bool, str]:
    raw = _normalize_spaces(value)
    numeric = re.sub(r"[^0-9.]", "", raw)
    if not numeric:
        return False, f"Invalid {label}. Please enter a numeric value."
    try:
        amount = float(numeric)
    except ValueError:
        return False, f"Invalid {label}. Please enter a numeric value."
    if amount <= 0:
        return False, f"Invalid {label}. Value must be greater than zero."
    return True, str(int(amount)) if amount.is_integer() else f"{amount:.2f}"


def validate_id_proof_number(value: str) -> Tuple[bool, str]:
    candidate = _normalize_spaces(value).upper().replace(" ", "")
    if len(candidate) < 4:
        return False, "ID proof number is too short."
    if not re.fullmatch(r"[A-Z0-9/-]+", candidate):
        return False, "ID proof number can contain letters, digits, / and - only."
    return True, candidate


def validate_address(value: str) -> Tuple[bool, str]:
    candidate = _normalize_spaces(value)
    if len(candidate) < 8:
        return False, "Address is too short. Please provide full address details."
    return True, candidate


def parse_skip(value: str) -> Optional[str]:
    normalized = _normalize_spaces(value).lower()
    return "skip" if normalized == "skip" else None
