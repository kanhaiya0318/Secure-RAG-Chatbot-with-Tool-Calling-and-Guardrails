from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.schemas.chat import ChatRequest, ChatResponse
from app.services.llm_service import LLMService
from app.services.chat_service import reply
from app.dependancies import get_db, get_llm_service, get_rag_pipeline  
from app.rag.pipeline import RAGPipeline


router = APIRouter(
    prefix="/chat",
    tags=["chat"]
)

@router.post("/query")
async def chat_stream(
    payload: ChatRequest,
    db:dict = Depends(get_db),
    llm_service:LLMService = Depends(get_llm_service),
    rag: RAGPipeline = Depends(get_rag_pipeline),
):
    session_id = payload.session_id
    message = payload.message

    if session_id not in db:
        db[session_id] = {"summary":"", "history":[]}

    return StreamingResponse(
        reply(db[session_id], message, llm_service, rag, 
            langsmith_extra={"metadata": {"session_id": session_id}}),
        media_type="text/plain",
    )
