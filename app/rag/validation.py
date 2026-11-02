import pymupdf


def validate_pdf(data: bytes, *, max_pages: int) -> int:
    """Check the bytes are a readable, unlocked PDF within the page limit. Returns the page count."""
    if not data:
        raise ValueError("The file is empty")
    if not data.startswith(b"%PDF-"):
        raise ValueError("The file is not a valid PDF")
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception:
        raise ValueError("The PDF is corrupted or cannot be read")
    try:
        if doc.needs_pass:
            raise ValueError("Password-protected PDFs are not supported")
        if doc.page_count > max_pages:
            raise ValueError(f"The PDF has {doc.page_count} pages; the limit is {max_pages}")
        return doc.page_count
    finally:
        doc.close()