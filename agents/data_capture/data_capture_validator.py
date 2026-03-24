import re
from datetime import datetime

def validate_name(value: str):
    value = value.strip()
    if len(value) < 3:
        return False, "Name too short."
    return True, value.upper()

def validate_pan(value: str):
    pattern = r"^[A-Z]{5}[0-9]{4}[A-Z]$"
    value = value.strip().upper()
    if not re.match(pattern, value):
        return False, "Invalid PAN format."
    return True, value

def validate_phone(value: str):
    value = re.sub(r"\D", "", value)
    if len(value) != 10:
        return False, "Invalid mobile number."
    return True, value

def validate_email(value: str):
    if "@" not in value:
        return False, "Invalid email address."
    return True, value.lower()

def validate_date(value: str):
    try:
        datetime.strptime(value, "%Y-%m-%d")
        return True, value
    except:
        return False, "Invalid date format (YYYY-MM-DD)."