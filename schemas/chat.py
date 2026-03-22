from pydantic import BaseModel

class ChatRequest(BaseModel):
    user_input : str
    session_id: str

class ChatResponse(BaseModel):
    message : str
    action : str | None = None
    completed : bool = False
    progress : int = 0
    step : int = 1
    current_main_step : str = "Detail Capture"