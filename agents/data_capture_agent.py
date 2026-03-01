import json 
from core.redis_client import redis_client
from validators.data_capture_validator import (
    validate_name,
    validate_pan,
    validate_phone,
    validate_email,
    validate_date
)

VALIDATOR_MAP = {
    "name": validate_name,
    "pan": validate_pan,
    "phone": validate_phone,
    "email": validate_email,
    "date": validate_date
}

SESSION_TTL = 3600 # 1 hour

STEP_CONFIG = {
    "account_type": {
        "question": "What type of account would you like to open?",
        "field_type": "select",
        "options": ["Savings", "Current"]
    },
    "full_name": {
        "question": "Please enter your full legal name as per PAN.",
        "field_type": "text",
        "validation": "name"
    },
    "dob": {
        "question": "Enter your date of birth (YYYY-MM-DD).",
        "field_type": "date",
        "validation": "date"
    },
    "pan": {
        "question": "Enter your PAN number.",
        "field_type": "text",
        "validation": "pan"
    },
    "phone": {
        "question": "Enter your mobile number linked to Aadhaar.",
        "field_type": "text",
        "validation": "phone"
    },
    "email": {
        "question": "Enter your email address.",
        "field_type": "email",
        "validation": "email"
    },
    "address": {
        "question": "Enter your residential address.",
        "field_type": "textarea"
    },
    "occupation": {
        "question": "What is your occupation?",
        "field_type": "text"
    },
    "digilocker_choice": {
        "question": "Would you like to fetch documents via DigiLocker?",
        "field_type": "select",
        "options": ["yes", "no"]
    },
    "confirmation": {
        "question": "Please confirm all details are correct.",
        "field_type": "select",
        "options": ["yes", "no"]
    }
}

steps = list(STEP_CONFIG.keys())

def calculate_progress(step_index):
    return int((step_index / len(steps)) * 100)


def get_key(session_id):
    return f"onboarding:{session_id}"

async def get_session(session_id):
    data = await redis_client.get(get_key(session_id))
    return json.loads(data) if data else {"step_index": 0, "data": {}}

async def save_session(session_id, session):
    await redis_client.set(
        get_key(session_id),
        json.dumps(session),
        ex=SESSION_TTL
    )

async def run(session_id: str, user_input: str):

    session = await get_session(session_id)
    step_index = session["step_index"]
    data = session["data"]

    # If first interaction
    if user_input == "":
        step_key = steps[0]
        step = STEP_CONFIG[step_key]

        return {
            "message": step["question"],
            "field_name": step_key,
            "field_type": step["field_type"],
            "options": step.get("options"),
            "validation": step.get("validation"),
            "progress": calculate_progress(0),
            "completed": False
        }

    # Current step
    step_key = steps[step_index]
    step = STEP_CONFIG[step_key]

    validation_type = step.get("validation")

    if validation_type:
        validator = VALIDATOR_MAP.get(validation_type)
        if validator:
            is_valid, result = validator(user_input)
            if not is_valid:
                return {
                    "message": result,
                    "field_name": step_key,
                    "field_type": step["field_type"],
                    "options": step.get("options"),
                    "validation": validation_type,
                    "progress": calculate_progress(step_index),
                    "completed": False
                }
            user_input = result  # sanitized value

    # Save sanitized value
    data[step_key] = user_input

    # Special DigiLocker branch
    if step_key == "digilocker_choice" and user_input.lower() == "yes":
        return {
            "message": "Redirecting to DigiLocker for authentication.",
            "action": "digilocker_auth",
            "progress": calculate_progress(step_index),
            "completed": False
        }

    # Move to next step
    step_index += 1

    # If onboarding complete
    if step_index >= len(steps):
        # TODO: Save to Postgres here
        await redis_client.delete(get_key(session_id))
        return {
            "message": "Onboarding data capture complete.",
            "progress": 100,
            "completed": True
        }

    # Save session
    session["step_index"] = step_index
    session["data"] = data
    await save_session(session_id, session)

    # Return next step
    next_step_key = steps[step_index]
    next_step = STEP_CONFIG[next_step_key]

    return {
        "message": next_step["question"],
        "field_name": next_step_key,
        "field_type": next_step["field_type"],
        "options": next_step.get("options"),
        "validation": next_step.get("validation"),
        "progress": calculate_progress(step_index),
        "completed": False
    }
    

