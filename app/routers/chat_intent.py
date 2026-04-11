"""
Chatbot Intent Classification Router
POST /api/v1/chat-intent
"""
from fastapi import APIRouter, Depends
from models.schemas import ChatIntentRequest, ChatIntentResponse
from app.dependencies.auth import verify_internal_token
from app.services.chat_intent import process_chat_intent

router = APIRouter(prefix="/api/v1")


@router.post("/chat-intent", response_model=ChatIntentResponse, dependencies=[Depends(verify_internal_token)])
async def chat_intent(request: ChatIntentRequest):
    """
    Chatbot mesajının niyetini sınıflandırır ve kişisel bilgileri maskeler.
    POST /api/v1/chat-intent
    { "message": "Bu sözleşmede cezai şart var mı?" }
    """
    result = await process_chat_intent(request.message)
    return ChatIntentResponse(**result)
