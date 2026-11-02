import re
import asyncio
import tempfile
from pathlib import Path
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, Request

from app.config import settings
from app.limits import upload_limiter, client_ip
from app.dependancies import get_rag_pipeline
from app.rag.pipeline import RAGPipeline
from app.schemas.rag import DeleteOut, DocumentOut, UploadOut
from app.rag.validation import validate_pdf


router = APIRouter(prefix="/rag", tags=["rag"])


_NAME_RE = re.compile(r"[^\w .()\-]")

def _clean_name(raw: str) -> str:
    name = _NAME_RE.sub("_", Path(raw or "").name)
    return name[: settings.MAX_FILENAME_CHARS] or "document.pdf"


@router.post("/upload", response_model=UploadOut)
async def upload(
    request: Request,
    file: UploadFile = File(...),
    rag: RAGPipeline = Depends(get_rag_pipeline),
):
    upload_limiter.hit(client_ip(request))

    raw_name = Path(file.filename or "").name
    if not raw_name.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")
    name = _clean_name(raw_name)

    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
    buf = bytearray()
    while block := await file.read(1 << 20):
        buf.extend(block)
        if len(buf) > max_bytes:
            raise HTTPException(413, f"File larger than {settings.MAX_UPLOAD_MB} MB")
    data = bytes(buf)

    try:
        await asyncio.to_thread(validate_pdf, data, max_pages=settings.MAX_PDF_PAGES)
    except ValueError as e:
        raise HTTPException(422, str(e))

    if len(rag.list_documents()) >= settings.MAX_DOCUMENTS:
        raise HTTPException(409, f"Document limit reached ({settings.MAX_DOCUMENTS}). Delete one first.")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "upload.pdf"
        path.write_bytes(data)
        try:
            return await rag.aingest(path, source_name=name)
        except ValueError as e:
            raise HTTPException(422, str(e))


@router.get("/documents", response_model=list[DocumentOut])
async def list_documents(rag: RAGPipeline=Depends(get_rag_pipeline)):
    return rag.list_documents()


@router.delete("/documents/{doc_id}", response_model=DeleteOut)
async def delete_document(doc_id: str, rag: RAGPipeline = Depends(get_rag_pipeline)):
    if not rag.delete_document(doc_id):
        raise HTTPException(404, "Document not found")
    return DeleteOut(doc_id=doc_id, deleted=True)