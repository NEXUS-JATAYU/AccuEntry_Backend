import os
from fastapi import FastAPI, Form, UploadFile, File, Form
import httpx
from fastapi.middleware.cors import CORSMiddleware
from schemas.chat import ChatRequest, ChatResponse
from orchestration.onboarding_workflow import handle_chat
from core.database import engine, Base
import models.customer_info

# Initialize DB tables
Base.metadata.create_all(bind=engine)

app = FastAPI()
ACCUVERIFY_URL = os.getenv("ACCUVERIFY_URL", "http://localhost:9000")
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

@app.post("/kyc/pan")
async def proxy_pan(session_id: str = Form(...), file: UploadFile = File(...)):
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{ACCUVERIFY_URL}/upload-pan",
            params={"user_id": session_id},          # AccuVerify expects user_id as query param
            files={"file": (file.filename, await file.read(), file.content_type)}
        )
    return resp.json()

@app.post("/kyc/aadhaar")
async def proxy_aadhaar(session_id: str = Form(...), file: UploadFile = File(...)):
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{ACCUVERIFY_URL}/upload-aadhaar",
            params={"user_id": session_id},
            files={"file": (file.filename, await file.read(), file.content_type)}
        )
    return resp.json()

@app.post("/kyc/selfie")
async def proxy_selfie(session_id: str = Form(...), file: UploadFile = File(...)):
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{ACCUVERIFY_URL}/upload-selfie",
            params={"user_id": session_id},
            files={"file": (file.filename, await file.read(), file.content_type)}
        )
    return resp.json()

@app.post("/kyc/approve")
async def proxy_approve(session_id: str = Form(...)):
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{ACCUVERIFY_URL}/agent/approve-kyc",
            params={"user_id": session_id, "agent_id": "accuentry-bot"}
        )
    return resp.json()
    
