import json
import datetime
import re
from pydantic import BaseModel, Field
from typing import Optional, Literal, Dict, Any, List, Annotated
from typing_extensions import TypedDict
import operator

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END

from core.redis_client import redis_client
from core.database import SessionLocal
from models.customer_info import CustomerDetails

from data_capture.data_capture_validator import (
    validate_name,
    validate_pan,
    validate_phone,
    validate_email,
    validate_date,
)

SESSION_TTL = 3600

FIELD_ORDER = [
    "account_type",
    "full_name",
    "dob",
    "pan",
    "phone",
    "email",
    "address",
    "occupation",
    "confirmation"
]

FIELD_QUESTIONS = {
    "account_type": "Enter account type (Savings/Current):",
    "full_name": "Enter full name:",
    "dob": "Enter DOB (YYYY-MM-DD):",
    "pan": "Enter PAN:",
    "phone": "Enter phone number:",
    "email": "Enter email:",
    "address": "Enter address:",
    "occupation": "Enter occupation:",
    "confirmation": "Confirm details? (yes/no):"
}

MAIN_STEPS = {
    "Detail Capture": [
        "account_type","full_name","dob","pan","phone","email","address","occupation","confirmation"
    ],
    "Identity Verification": [
        "kyc_verification", "ocr_recognition", "live_kyc"
    ]
}

def current_main_step(extracted_data, missing_fields):
    for main, subs in MAIN_STEPS.items():
        for field in subs:
            if field in missing_fields:
                return main
    return "Completed"

# -----------------------------------------
# Schema
# -----------------------------------------
class OnboardingData(BaseModel):
    account_type: Optional[Literal["Savings", "Current"]] = None
    full_name: Optional[str] = None
    dob: Optional[str] = None
    pan: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    address: Optional[str] = None
    occupation: Optional[str] = None
    confirmation: Optional[Literal["yes", "no"]] = None

REQUIRED_FIELDS = list(OnboardingData.model_fields.keys())

# -----------------------------------------
# State - IMPORTANT: MUST be TypedDict for LangGraph to reduce properly
# -----------------------------------------
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    extracted_data: dict
    current_missing_field: Optional[str]
    validation_error: Optional[str]
    onboarding_complete: Optional[bool]
    temp_input: Optional[str]

# -----------------------------------------
# LLM
# -----------------------------------------
llm = ChatOllama(model="gemma2:2b", temperature=0)

# -----------------------------------------
# Helpers
# -----------------------------------------
def calculate_progress(data):
    filled = sum(1 for k in REQUIRED_FIELDS if data.get(k))
    return int((filled / len(REQUIRED_FIELDS)) * 100)

def get_key(session_id):
    return f"onboarding_lg:{session_id}"

# -----------------------------------------
# Redis Session
# -----------------------------------------
async def get_session(session_id) -> AgentState:
    data = await redis_client.get(get_key(session_id))
    if data:
        parsed = json.loads(data)
        return {
            "messages": [
                HumanMessage(content=m["content"]) if m["type"] == "user" else AIMessage(content=m["content"])
                for m in parsed["messages"]
            ],
            "extracted_data": parsed["extracted_data"],
            "current_missing_field": parsed.get("current_missing_field"),
            "validation_error": parsed.get("validation_error"),
            "onboarding_complete": False,
            "temp_input": None
        }

    return {
        "messages": [],
        "extracted_data": {},
        "current_missing_field": None,
        "validation_error": None,
        "onboarding_complete": False,
        "temp_input": None
    }

async def save_session(session_id, state: AgentState):
    payload = {
        "messages": [
            {"type": "user" if isinstance(m, HumanMessage) else "ai", "content": m.content}
            for m in state["messages"]
        ],
        "extracted_data": state["extracted_data"],
        "current_missing_field": state.get("current_missing_field"),
        "validation_error": state.get("validation_error"),
    }
    await redis_client.set(get_key(session_id), json.dumps(payload), ex=SESSION_TTL)

# -----------------------------------------
# Nodes
# -----------------------------------------
def input_node(state: AgentState):
    if not state["messages"]:
        return {}

    user_input = state["messages"][-1].content.strip()
    current_field = state.get("current_missing_field")

    # First step
    if not current_field:
        return {
            "current_missing_field": FIELD_ORDER[0],
            "temp_input": user_input  
        }

    return {"temp_input": user_input}

def validation_node(state: AgentState):
    data = dict(state.get("extracted_data", {}))
    field = state.get("current_missing_field")
    value = state.get("temp_input")

    if not field or value is None:
        return {"validation_error": None}

    # Intelligent Extraction for Account Type
    if field == "account_type":
        val = value.lower()
        if "saving" in val:
            data[field] = "Savings"
            return {"extracted_data": data, "validation_error": None, "temp_input": None}
        elif "current" in val or "checking" in val:
            data[field] = "Current"
            return {"extracted_data": data, "validation_error": None, "temp_input": None}
        else:
            return {"validation_error": "Please choose Savings or Current account."}

    # Confirmation
    elif field == "confirmation":
        if "yes" in value.lower():
            data[field] = "yes"
            return {"extracted_data": data, "validation_error": None, "temp_input": None}
        elif "no" in value.lower():
            return {"validation_error": "You said no. Please confirm with 'yes' when you are happy with the details, or we can restart the process."}
        else:
            return {"validation_error": "Please enter 'yes' to confirm or 'no'."}

    # Validators
    else:
        validators = {
            "full_name": validate_name,
            "pan": validate_pan,
            "phone": validate_phone,
            "email": validate_email,
            "dob": validate_date,
        }

        if field in validators:
            is_valid, result = validators[field](value)
            if not is_valid:
                return {"validation_error": result}
            data[field] = result
        else:
            data[field] = value

    return {"extracted_data": data, "validation_error": None, "temp_input": None}

def response_node(state: AgentState):
    data = state["extracted_data"]
    field = state.get("current_missing_field")
    error = state.get("validation_error")

    if error:
        msg = error
        return {"messages": [AIMessage(content=msg)]}

   
    if field == "confirmation" and data.get("confirmation") == "yes":
        msg = "Onboarding complete"
        return {"current_missing_field": None, "onboarding_complete": True, "messages": [AIMessage(content=msg)]}

    # Move to next field
    current_index = FIELD_ORDER.index(field)
    if current_index + 1 < len(FIELD_ORDER):
        next_field = FIELD_ORDER[current_index + 1]
        msg = FIELD_QUESTIONS[next_field]
        
        # Override the msg if the next field is confirmation
        if next_field == "confirmation":
            details = "\n".join([f"• {k.replace('_', ' ').capitalize()}: {v}" for k, v in data.items() if k != "confirmation"])
            msg = f"Please confirm your details are correct:\n\n{details}\n\nType 'yes' to confirm and save."

        return {"current_missing_field": next_field, "messages": [AIMessage(content=msg)]}
    else:
        msg = "Onboarding complete"
        return {"current_missing_field": None, "onboarding_complete": True, "messages": [AIMessage(content=msg)]}


# -----------------------------------------
# Graph
# -----------------------------------------
workflow = StateGraph(AgentState)
workflow.add_node("input", input_node)
workflow.add_node("validate", validation_node)
workflow.add_node("response", response_node)

workflow.add_edge(START, "input")
workflow.add_edge("input", "validate")
workflow.add_edge("validate", "response")
workflow.add_edge("response", END)

app = workflow.compile()

# -----------------------------------------
# Main Run
# -----------------------------------------
async def run(session_id: str, user_input: str):
    state = await get_session(session_id)

    # Convert initial empty string requests from frontend (when widget opens) to help message
    if user_input.strip():
        state["messages"].append(HumanMessage(content=user_input))
    new_state = await app.ainvoke(state)
    await save_session(session_id, new_state)

    ai_msg = new_state["messages"][-1].content
    data = new_state["extracted_data"]

    progress = calculate_progress(data)
    missing = [f for f in REQUIRED_FIELDS if not data.get(f)]

    #Save to DB
    if new_state.get("onboarding_complete"):
        required_check = all([
            data.get("account_type"),
            data.get("full_name"),
            data.get("dob"),
            data.get("pan"),
            data.get("phone"),
            data.get("email"),
            data.get("address"),
            data.get("occupation"),
            data.get("confirmation") == "yes"
        ])

        if not required_check:
            return {
                "message": "Incomplete data",
                "progress": progress,
                "completed": False
            }

        db = SessionLocal()

        try:
            customer = CustomerDetails(
                c_name=data["full_name"],
                c_account_type=data["account_type"],
                c_phone_number=data["phone"],
                c_email=data["email"],
                c_address=data["address"],
                c_occupation=data["occupation"],
                c_pan=data["pan"],
                c_dob=datetime.datetime.strptime(data["dob"], "%Y-%m-%d").date()
            )

            db.add(customer)
            db.commit()

            await redis_client.delete(get_key(session_id))

            return {
                "message": "Onboarding complete! Your data has been saved successfully. We will now proceed to verification.",
                "progress": 100,
                "completed": True,
            }

        except Exception as e:
            print("DB Error:", e)
            db.rollback()

            return {
                "message": f"Database error: {e}",
                "progress": progress,
                "completed": False
            }

        finally:
            db.close()

    return {
      "message": ai_msg,
      "progress": progress,
      "completed": new_state.get("onboarding_complete", False),
      "missing_fields": missing,
      "main_steps": MAIN_STEPS,
      "current_main_step": current_main_step(data, missing)
    }
