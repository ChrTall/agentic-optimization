import io

import pytest
from pypdf import PdfWriter

from agentic_optimization.ui.pdf_input import (
    MAX_PDF_SIZE_BYTES,
    extract_text_from_pdf,
)


def _build_pdf_with_text(text: str) -> io.BytesIO:
    """Erzeugt ein minimales PDF mit einer Textzeile für Tests."""

    reportlab = pytest.importorskip("reportlab.pdfgen.canvas")

    buffer = io.BytesIO()
    canvas = reportlab.Canvas(buffer)
    canvas.drawString(72, 720, text)
    canvas.save()
    buffer.seek(0)
    return buffer


def _build_empty_pdf() -> io.BytesIO:
    """Erzeugt ein gültiges PDF ohne jeglichen Text (wie ein reines Scan-PDF)."""

    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)

    buffer = io.BytesIO()
    writer.write(buffer)
    buffer.seek(0)
    return buffer


def test_extract_text_from_pdf_returns_text() -> None:
    pdf_file = _build_pdf_with_text("Liefervertrag Testinhalt")

    result = extract_text_from_pdf(pdf_file)

    assert result.error is None
    assert "Liefervertrag Testinhalt" in result.text
    assert result.page_count == 1


def test_extract_text_from_pdf_without_text_returns_error() -> None:
    pdf_file = _build_empty_pdf()

    result = extract_text_from_pdf(pdf_file)

    assert result.text == ""
    assert result.error is not None
    assert result.page_count == 1


def test_extract_text_from_pdf_rejects_invalid_file() -> None:
    invalid_file = io.BytesIO(b"das ist kein pdf")

    result = extract_text_from_pdf(invalid_file)

    assert result.text == ""
    assert result.error is not None
    assert result.page_count == 0


def test_extract_text_from_pdf_rejects_oversized_file() -> None:
    pdf_file = _build_pdf_with_text("Kurzer Text")
    pdf_file.size = MAX_PDF_SIZE_BYTES + 1

    result = extract_text_from_pdf(pdf_file, max_size_bytes=MAX_PDF_SIZE_BYTES)

    assert result.text == ""
    assert result.error is not None
    assert "zu groß" in result.error