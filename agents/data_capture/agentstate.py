from pydantic import BaseModel
from typing import TypedDict

class AgentState(TypedDict):
    messages: list
    full_name: str | None
    dob: str | None
    pan: str | None
    phone: str | None
    email: str | None
    account_type: str | None 