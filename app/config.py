from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal

class Settings(BaseSettings):
    GEMINI_API_KEY : str
    GEMINI_MODEL : str = "gemini-3.1-flash-lite"
    GEMINI_EMBED_MODEL : str 

    RETRY_LLM_CALL : int = 5
    TRIGGER : int = 8
    KEEP_LAST : int = 4

    #web search
    SEARCH_DEPTH : Literal["advanced", "ultra-fast", "fast", "basic"] = "advanced"
    SEARCH_MAX_RESULTS: int = 5 
    TAVILY_API_KEY : str

    #build rag
    CHUNK_SIZE : int = 700
    CHUNK_OVERLAP : int = 100
    OUTPUT_DIM : int = 1024
    BATCH_SIZE : int = 10

    #limits(Guardrails)
    CHAT_PER_MIN_IP : int = 30
    CHAT_PER_MIN_SESSION : int = 10
    UPLOADS_PER_10MIN : int = 5
    MAX_SESSIONS : int = 200
    SESSION_TTL_SECONDS : int = 3600
    MAX_TOOL_CALLS_PER_TURN : int = 3
    MAX_QUERY_CHARS : int = 300

    #uploads(Guardrails)
    MAX_UPLOAD_MB : int = 20
    MAX_PDF_PAGES : int = 100
    MAX_CHUNKS_PER_DOC : int = 800
    MAX_DOCUMENTS : int = 20
    MAX_FILENAME_CHARS : int = 80

    #answer cache
    ANSWER_CACHE_ENABLED : bool = True
    ANSWER_CACHE_MAX_ENTRIES : int = 200
    ANSWER_CACHE_TTL_SECONDS : int = 3600

    model_config = SettingsConfigDict(
        env_file = ".env",
        env_file_encoding = "utf-8",
        extra = "ignore"
    )


settings = Settings()