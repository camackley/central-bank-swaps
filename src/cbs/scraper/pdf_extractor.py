"""PDF press release text extraction — PyMuPDF primary, pdfplumber fallback."""

from __future__ import annotations

from pathlib import Path

import pdfplumber
import pymupdf
from pydantic import BaseModel

CHUNK_PAGE_LIMIT = 15


class PDFExtractionError(Exception):
    """Raised when text extraction fails for both backends."""


class PDFChunk(BaseModel):
    """A chunk of extracted text covering a range of pages."""

    text: str
    start_page: int
    end_page: int


class PDFExtractResult(BaseModel):
    """Result of extracting text from a PDF file."""

    text: str
    page_count: int
    chunks: list[PDFChunk]
    extractor: str


def _extract_with_pymupdf(path: Path) -> tuple[list[str], int]:
    """Extract per-page text using PyMuPDF.

    Returns:
        Tuple of (list of per-page text strings, page count).
    """
    doc = pymupdf.open(str(path))  # type: ignore[no-untyped-call]
    try:
        pages: list[str] = [page.get_text() for page in doc]  # type: ignore[attr-defined]
        return pages, len(doc)
    finally:
        doc.close()  # type: ignore[no-untyped-call]


def _extract_with_pdfplumber(path: Path) -> tuple[list[str], int]:
    """Extract per-page text using pdfplumber.

    Returns:
        Tuple of (list of per-page text strings, page count).
    """
    pdf = pdfplumber.open(str(path))
    try:
        pages = [page.extract_text() or "" for page in pdf.pages]
        return pages, len(pdf.pages)
    finally:
        pdf.close()


def _build_chunks(pages: list[str]) -> list[PDFChunk]:
    """Group per-page text into chunks of up to CHUNK_PAGE_LIMIT pages."""
    chunks: list[PDFChunk] = []
    for i in range(0, len(pages), CHUNK_PAGE_LIMIT):
        batch = pages[i : i + CHUNK_PAGE_LIMIT]
        start_page = i + 1  # 1-indexed
        end_page = i + len(batch)
        text = "\n".join(batch)
        chunks.append(PDFChunk(text=text, start_page=start_page, end_page=end_page))
    return chunks


def extract_pdf(path: Path) -> PDFExtractResult:
    """Extract text from a PDF file.

    Uses PyMuPDF as the primary backend, falling back to pdfplumber on failure.
    Long PDFs (>CHUNK_PAGE_LIMIT pages) are split into chunks.

    Raises:
        FileNotFoundError: If the PDF file does not exist.
        PDFExtractionError: If both backends fail to extract text.
    """
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")

    extractor = "pymupdf"
    try:
        pages, page_count = _extract_with_pymupdf(path)
    except Exception:
        try:
            pages, page_count = _extract_with_pdfplumber(path)
            extractor = "pdfplumber"
        except Exception as exc:
            msg = f"Failed to extract text from {path}: {exc}"
            raise PDFExtractionError(msg) from exc

    chunks = _build_chunks(pages)
    full_text = "\n".join(c.text for c in chunks)

    return PDFExtractResult(
        text=full_text,
        page_count=page_count,
        chunks=chunks,
        extractor=extractor,
    )
