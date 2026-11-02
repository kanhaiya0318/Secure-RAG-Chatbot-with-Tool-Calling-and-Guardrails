from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from app.schemas.chat import ChatRequest, ChatResponse
from app.services.llm_service import LLMService
from app.services.chat_service import reply
from app.dependancies import get_db, get_llm_service, get_rag_pipeline  
from app.rag.pipeline import RAGPipeline
from app.limits import chat_ip_limiter, chat_session_limiter, client_ip, get_or_create_session


router = APIRouter(
    prefix="/chat",
    tags=["chat"]
)

@router.post("/query")
async def chat_stream(
    request: Request,
    payload: ChatRequest,
    db:dict = Depends(get_db),
    llm_service:LLMService = Depends(get_llm_service),
    rag: RAGPipeline = Depends(get_rag_pipeline),
):

    chat_ip_limiter.hit(client_ip(request))
    chat_session_limiter.hit(payload.session_id)
    session = get_or_create_session(db, payload.session_id)

    return StreamingResponse(
        reply(session, payload.message, llm_service, rag, 
            langsmith_extra={"metadata": {"session_id": payload.session_id}}),
        media_type="text/plain",
    )
