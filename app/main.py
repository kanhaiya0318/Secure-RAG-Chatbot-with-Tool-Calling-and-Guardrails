from dotenv import load_dotenv
load_dotenv()

import time
import httpx
from fastapi import FastAPI
from contextlib import asynccontextmanager

from app.utils.timing import logger
from app.routers.sessions import router as session_router
from app.routers.chat import router as chat_router
from app.routers.rag import router as rag_router
from app.dependancies import build_rag_pipeline



@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.gemini = httpx.AsyncClient(
        base_url= "",
        timeout = httpx.Timeout(60.0, connect = 20.0),
        limits= httpx.Limits(max_connections=3, max_keepalive_connections=2, keepalive_expiry=20.0),
    )
    app.state.rag = build_rag_pipeline()    
    yield
    await app.state.gemini.aclose()



app = FastAPI(lifespan = lifespan)

app.include_router(session_router)
app.include_router(chat_router)
app.include_router(rag_router)

@app.middleware("http")
async def timing_middleware(request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - start)*1000
    response.headers["X-Process-Time-ms"] = f"{ms:.0f}"
    logger.info(f"[Time] {request.method} {request.url.path} total: {ms:.0f} ms")
    return response

@app.get("/")
async def root():
    return {"message": "Doing Great"}



