from pydantic import BaseModel

class ChatRequest(BaseModel):
    user_input : str
    session_id: str

class ChatResponse(BaseModel):
    message: str
    progress: int = 0
    requires_upload: bool = False
    stage: str = "data_capture"
    action: str | None = None
    completed: bool = False
    step: int = 1
    current_main_step: str = "Detail Capture"
    aml_status: str | None = "pending"
    aml_in_background: bool = False