from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from schemas.chat import ChatRequest, ChatResponse
from orchestration.onboarding_workflow import handle_chat



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

@app.post("/chat",response_model=ChatResponse)
async def chat_endpoint(request : ChatRequest):
    try:
        response = await handle_chat (
            request.session_id,
            request.user_input
        )
        return response 
    except Exception as e:
        print(f"Error handling chat: {e}")
        raise e 