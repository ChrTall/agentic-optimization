"""
Lässt ein LLM den Text eines hochgeladenen Dokuments (z. B. Liefervertrag)
nach konkreten betrieblichen Kennzahlen durchsuchen (Preise, Kosten,
Mengen, Kapazitäten, Lieferzeiten usw.).

Ergänzt den requirement_agent: Dieser erkennt die Problemstruktur
(Entscheidung, Ziel, Constraints als Freitext); dieses Modul zieht
zusätzlich konkrete Zahlenwerte samt Beleg-Textstelle aus dem Dokument.
"""

import json
import re
from typing import TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage


class ExtractedParameter(TypedDict):
    """Ein einzelner, aus dem Dokument erkannter Wert."""

    label: str
    value: str
    unit: str | None
    evidence: str | None


DOCUMENT_PARAMETER_SYSTEM_PROMPT = """Du analysierst den Text eines
hochgeladenen Geschäftsdokuments (z. B. Liefervertrag, Spezifikation,
Planungsrichtlinie).

Deine Aufgabe: Finde alle konkreten betrieblichen Kennzahlen und
Parameter, die für ein Optimierungsmodell relevant sein könnten.
Typische Beispiele:
- Preise, Kosten, Rabatte
- Mengen (Mindestbestellmenge, Höchstmenge, Losgröße)
- Kapazitäten (Lager, Produktion)
- Lieferzeiten, Vertragslaufzeiten, Fristen
- Servicelevel- oder Qualitätsvorgaben

Gib deine Antwort ausschließlich als JSON-Array zurück. Verwende keine
Markdown-Backticks und keine zusätzlichen Erklärungen.

Format:
[
  {
    "label": "Stückpreis",
    "value": "4.50",
    "unit": "€",
    "evidence": "Der Preis pro Einheit beträgt 4,50 €."
  }
]

Regeln:
- Erfinde keine Werte. Nimm ausschließlich Angaben auf, die wörtlich
  oder eindeutig im Text stehen.
- "value" ist immer ein möglichst einfacher, direkt weiterverwendbarer
  Wert (Zahl als String, oder kurzer Text bei nicht-numerischen
  Angaben).
- "unit" ist die Einheit, falls vorhanden, sonst null.
- "evidence" ist ein kurzes, nahezu wörtliches Zitat aus dem Text
  (maximal ein Satz), das den Wert belegt.
- Gib ein leeres Array zurück, wenn keine konkreten Werte gefunden
  wurden.
"""


def extract_document_parameters(
    llm: BaseChatModel,
    document_text: str,
) -> list[ExtractedParameter]:
    """
    Sucht per LLM nach konkreten Kennzahlen im Dokumenttext.

    Gibt bei leerem Text, technischen Fehlern oder einer nicht
    auswertbaren LLM-Antwort eine leere Liste zurück, statt eine
    Exception zu werfen. Diese Funktion darf den Analyse-Flow niemals
    blockieren.
    """

    if not document_text.strip():
        return []

    try:
        response = llm.invoke([
            SystemMessage(
                content=DOCUMENT_PARAMETER_SYSTEM_PROMPT
            ),
            HumanMessage(content=document_text.strip()),
        ])
        parsed = _parse_json_array(str(response.content))
    except Exception:
        return []

    return [
        _normalize_extracted_parameter(item)
        for item in parsed
        if isinstance(item, dict)
    ]


def format_parameter_display(
    parameter: ExtractedParameter,
) -> str:
    """Formatiert einen erkannten Wert für Anzeige und Übernahme in den Freitext."""

    value_text = parameter["value"]

    if parameter.get("unit"):
        value_text = f"{value_text} {parameter['unit']}"

    return f"{parameter['label']}: {value_text}"


def _parse_json_array(text: str) -> list:
    clean = re.sub(r"```(?:json)?|```", "", text).strip()
    parsed = json.loads(clean)

    if not isinstance(parsed, list):
        raise TypeError(
            "Die LLM-Antwort muss ein JSON-Array sein."
        )

    return parsed


def _normalize_extracted_parameter(
    item: dict,
) -> ExtractedParameter:
    unit = item.get("unit")
    evidence = item.get("evidence")

    return {
        "label": (
            str(item.get("label", "")).strip()
            or "Unbenannter Wert"
        ),
        "value": str(item.get("value", "")).strip(),
        "unit": str(unit).strip() if unit else None,
        "evidence": (
            str(evidence).strip() if evidence else None
        ),
    }