from pydantic import BaseModel

class ChatRequest(BaseModel):
    user_input : str
    session_id: str

class ChatResponse(BaseModel):
    message : str
    action : str | None = None
    completed : bool =False