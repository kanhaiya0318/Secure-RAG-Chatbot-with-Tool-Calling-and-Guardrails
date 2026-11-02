import tempfile
from pathlib import Path
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, Request

from app.limits import upload_limiter, client_ip
from app.dependancies import get_rag_pipeline
from app.rag.pipeline import RAGPipeline
from app.schemas.rag import DeleteOut, DocumentOut, UploadOut


router = APIRouter(prefix="/rag", tags=["rag"])

MAX_UPLOAD_MB = 20


@router.post("/upload", response_model=UploadOut)
async def upload(request: Request, file: UploadFile = File(...), rag:RAGPipeline = Depends(get_rag_pipeline)):
    upload_limiter.hit(client_ip(request))
    name = Path(file.filename or "").name
    if not name.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")

    data = await file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File larger than {MAX_UPLOAD_MB} MB")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "upload.pdf"
        path.write_bytes(data)
        try:
            result = await rag.aingest(path, source_name=name)
        except ValueError as e:
            raise HTTPException(422, str(e))
        return result


@router.get("/documents", response_model=list[DocumentOut])
async def list_documents(rag: RAGPipeline=Depends(get_rag_pipeline)):
    return rag.list_documents()


@router.delete("/documents/{doc_id}", response_model=DeleteOut)
async def delete_document(doc_id: str, rag: RAGPipeline = Depends(get_rag_pipeline)):
    if not rag.delete_document(doc_id):
        raise HTTPException(404, "Document not found")
    return DeleteOut(doc_id=doc_id, deleted=True)