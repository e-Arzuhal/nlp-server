"""
Chatbot Intent Classification Router
POST /api/v1/chat-intent
"""
import logging

from fastapi import APIRouter, Depends
from models.schemas import ChatIntentRequest, ChatIntentResponse
from app.dependencies.auth import verify_internal_token
from app.services.chat_intent import process_chat_intent

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1")


@router.post("/chat-intent", response_model=ChatIntentResponse, dependencies=[Depends(verify_internal_token)])
async def chat_intent(request: ChatIntentRequest):
    """
    Chatbot mesajının niyetini sınıflandırır ve kişisel bilgileri maskeler.
    POST /api/v1/chat-intent
    { "message": "Bu sözleşmede cezai şart var mı?" }
    """
    logger.debug("chat_intent_request", extra={"message_length": len(request.message)})
    result = await process_chat_intent(request.message)
    logger.info("chat_intent_response", extra={
        "intent": result["intent"],
        "confidence": result["confidence"],
    })
    return ChatIntentResponse(**result)
