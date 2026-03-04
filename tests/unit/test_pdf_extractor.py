"""Tests for PDF press release text extraction (Slice 1.8)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from cbs.scraper.pdf_extractor import (
    PDFExtractionError,
    PDFExtractResult,
    extract_pdf,
)

FIXTURES = Path(__file__).parent.parent / "fixtures" / "pdf"


class TestExtractTextFromShortPdf:
    """Test text extraction from a short (2-page) PDF press release."""

    def test_extracts_text_from_short_pdf(self) -> None:
        result = extract_pdf(FIXTURES / "short_press_release.pdf")
        assert isinstance(result, PDFExtractResult)
        assert "Federal Reserve Board" in result.text
        assert "swap arrangements" in result.text

    def test_result_includes_page_count(self) -> None:
        result = extract_pdf(FIXTURES / "short_press_release.pdf")
        assert result.page_count == 2

    def test_result_text_is_nonempty(self) -> None:
        result = extract_pdf(FIXTURES / "short_press_release.pdf")
        assert len(result.text.strip()) > 0

    def test_result_includes_extractor_used(self) -> None:
        result = extract_pdf(FIXTURES / "short_press_release.pdf")
        assert result.extractor in ("pymupdf", "pdfplumber")


class TestExtractTextFromLongPdfWithChunking:
    """Test text extraction from a long (20-page) PDF with chunking."""

    def test_extracts_text_from_long_pdf(self) -> None:
        result = extract_pdf(FIXTURES / "long_report.pdf")
        assert isinstance(result, PDFExtractResult)
        assert result.page_count == 20

    def test_long_pdf_returns_chunks(self) -> None:
        result = extract_pdf(FIXTURES / "long_report.pdf")
        assert len(result.chunks) > 1

    def test_each_chunk_has_page_range(self) -> None:
        result = extract_pdf(FIXTURES / "long_report.pdf")
        for chunk in result.chunks:
            assert chunk.start_page >= 1
            assert chunk.end_page >= chunk.start_page

    def test_chunks_cover_all_pages(self) -> None:
        result = extract_pdf(FIXTURES / "long_report.pdf")
        covered_pages = set()
        for chunk in result.chunks:
            covered_pages.update(range(chunk.start_page, chunk.end_page + 1))
        assert covered_pages == set(range(1, 21))

    def test_short_pdf_returns_single_chunk(self) -> None:
        result = extract_pdf(FIXTURES / "short_press_release.pdf")
        assert len(result.chunks) == 1

    def test_full_text_equals_joined_chunks(self) -> None:
        result = extract_pdf(FIXTURES / "long_report.pdf")
        joined = "\n".join(c.text for c in result.chunks)
        assert result.text == joined


class TestPdfFallbackChain:
    """Test that PyMuPDF failure triggers pdfplumber fallback."""

    def test_uses_pymupdf_by_default(self) -> None:
        result = extract_pdf(FIXTURES / "short_press_release.pdf")
        assert result.extractor == "pymupdf"

    def test_falls_back_to_pdfplumber_when_pymupdf_fails(self) -> None:
        with patch(
            "cbs.scraper.pdf_extractor._extract_with_pymupdf",
            side_effect=Exception("PyMuPDF failed"),
        ):
            result = extract_pdf(FIXTURES / "short_press_release.pdf")
            assert result.extractor == "pdfplumber"
            assert "Federal Reserve Board" in result.text

    def test_fallback_result_still_has_page_count(self) -> None:
        with patch(
            "cbs.scraper.pdf_extractor._extract_with_pymupdf",
            side_effect=Exception("PyMuPDF failed"),
        ):
            result = extract_pdf(FIXTURES / "short_press_release.pdf")
            assert result.page_count == 2


class TestCorruptPdfRaisesClearError:
    """Test that corrupt/unreadable PDFs raise a clear error."""

    def test_corrupt_pdf_raises_extraction_error(self) -> None:
        with pytest.raises(PDFExtractionError, match="[Ff]ailed.*extract"):
            extract_pdf(FIXTURES / "corrupt.pdf")

    def test_nonexistent_pdf_raises_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            extract_pdf(FIXTURES / "nonexistent.pdf")

    def test_error_includes_file_path(self) -> None:
        with pytest.raises(PDFExtractionError) as exc_info:
            extract_pdf(FIXTURES / "corrupt.pdf")
        assert "corrupt.pdf" in str(exc_info.value)
