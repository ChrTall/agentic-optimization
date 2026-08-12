"""
Extrahiert Fließtext aus hochgeladenen PDF-Dokumenten (z. B. Lieferverträge),
damit dieser Text wie eine normale Freitext-Problembeschreibung in den
bestehenden Workflow (raw_input) eingespeist werden kann.

Dieses Modul kennt Streamlit bewusst nicht, damit die Extraktionslogik
unabhängig von der UI getestet werden kann.
"""

from dataclasses import dataclass
from typing import BinaryIO

from pypdf import PdfReader
from pypdf.errors import PdfReadError

MAX_PDF_SIZE_BYTES = 15 * 1024 * 1024  # 15 MB


@dataclass(frozen=True)
class PdfExtractionResult:
    """
    Ergebnis einer PDF-Textextraktion.

    text:
        Der extrahierte Fließtext. Leer, wenn nichts extrahiert werden konnte.

    page_count:
        Anzahl der Seiten des PDFs (0, wenn das PDF nicht gelesen werden konnte).

    error:
        Verständliche Fehlermeldung für die UI, falls die Extraktion
        fehlgeschlagen ist oder kein Text gefunden wurde. None bei Erfolg.
    """

    text: str
    page_count: int
    error: str | None


def extract_text_from_pdf(
    file: BinaryIO,
    *,
    max_size_bytes: int = MAX_PDF_SIZE_BYTES,
) -> PdfExtractionResult:
    """
    Liest ein PDF (z. B. einen Liefervertrag) ein und extrahiert den
    enthaltenen Text seitenweise.

    Gibt bei technischen Fehlern oder fehlendem Text (z. B. bei rein
    gescannten PDFs ohne Texterkennung) ein Ergebnis mit leerem Text und
    einer erklärenden Fehlermeldung zurück, statt eine Exception zu werfen.
    Die aufrufende UI-Schicht kann diese Meldung direkt anzeigen.
    """

    size = getattr(file, "size", None)

    if size is not None and size > max_size_bytes:
        return PdfExtractionResult(
            text="",
            page_count=0,
            error=(
                "Die PDF-Datei ist zu groß (maximal "
                f"{max_size_bytes // (1024 * 1024)} MB)."
            ),
        )

    try:
        reader = PdfReader(file)
    except PdfReadError:
        return PdfExtractionResult(
            text="",
            page_count=0,
            error=(
                "Die Datei konnte nicht als PDF gelesen werden. "
                "Bitte prüfen Sie, ob es sich um ein gültiges, "
                "unverschlüsseltes PDF handelt."
            ),
        )

    if reader.is_encrypted:
        return PdfExtractionResult(
            text="",
            page_count=0,
            error=(
                "Das PDF ist passwortgeschützt und kann nicht "
                "automatisch gelesen werden."
            ),
        )

    page_texts: list[str] = []

    for page in reader.pages:
        page_text = page.extract_text() or ""
        page_text = page_text.strip()

        if page_text:
            page_texts.append(page_text)

    extracted_text = "\n\n".join(page_texts).strip()

    if not extracted_text:
        return PdfExtractionResult(
            text="",
            page_count=len(reader.pages),
            error=(
                "Aus dem PDF konnte kein Text extrahiert werden. "
                "Vermutlich handelt es sich um ein gescanntes Dokument "
                "ohne Texterkennung (OCR). Bitte fügen Sie den Text "
                "stattdessen manuell ein."
            ),
        )

    return PdfExtractionResult(
        text=extracted_text,
        page_count=len(reader.pages),
        error=None,
    )