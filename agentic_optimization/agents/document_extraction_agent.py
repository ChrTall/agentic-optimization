"""
Lässt ein LLM den Text eines hochgeladenen Dokuments (z. B. Liefervertrag)
nach konkreten betrieblichen Kennzahlen durchsuchen (Preise, Kosten,
Mengen, Kapazitäten, Lieferzeiten usw.).

Ergänzt den requirement_agent: Dieser erkennt die Problemstruktur
(Entscheidung, Ziel, Constraints als Freitext); dieses Modul zieht
zusätzlich konkrete Zahlenwerte samt Beleg-Textstelle aus dem Dokument.

Zusätzlich implementiert dieses Modul eine vereinfachte „Decision-first
Extraction": Zuerst wird die vermutliche Entscheidung/Zielsetzung
erkannt, danach wird -- falls eine passende Modellvariante gefunden
wird -- gezielt nur nach deren benötigten Parametern gesucht, statt
nach beliebigen Werten im Dokument.
"""

import json
import re
from typing import TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from agentic_optimization.agents.requirement_agent import (
    identify_requirements_from_text,
)
from agentic_optimization.models.registry import (
    find_matching_model_variant,
)


class ExtractedParameter(TypedDict):
    """Ein einzelner, aus dem Dokument erkannter Wert."""

    field_name: str | None
    label: str
    value: str
    unit: str | None
    confidence: float | None
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

    display = f"{parameter['label']}: {value_text}"

    confidence = parameter.get("confidence")

    if confidence is not None and confidence < 0.7:
        display = f"{display} ⚠️"

    return display


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
    confidence = item.get("confidence")
    field_name = item.get("field_name")

    return {
        "field_name": (
            str(field_name).strip() if field_name else None
        ),
        "label": (
            str(item.get("label", "")).strip()
            or "Unbenannter Wert"
        ),
        "value": str(item.get("value", "")).strip(),
        "unit": str(unit).strip() if unit else None,
        "confidence": (
            float(confidence)
            if isinstance(confidence, (int, float))
            else None
        ),
        "evidence": (
            str(evidence).strip() if evidence else None
        ),
    }


def build_field_specs_from_variant(
    variant_definition: dict,
) -> list[dict]:
    """
    Baut aus einer erkannten Modellvariante die Liste der Parameter,
    nach denen gezielt im Dokument gesucht werden soll: die
    Eingabefelder des Modells selbst plus die Parameter aller von
    dieser Variante unterstützten Constraints.
    """

    field_specs = [
        {
            "name": field["name"],
            "label": field["label"],
            "unit": field.get("unit"),
        }
        for field in variant_definition["input_fields"]
    ]

    for constraint_def in variant_definition[
        "supported_constraints"
    ].values():
        for parameter in constraint_def["parameters"]:
            field_specs.append({
                "name": (
                    f"{constraint_def['type']}."
                    f"{parameter['name']}"
                ),
                "label": (
                    f"{constraint_def['name']} – "
                    f"{parameter['label']}"
                ),
                "unit": parameter.get("unit"),
            })

    return field_specs


def _build_targeted_extraction_prompt(
    field_specs: list[dict],
) -> str:
    field_lines = "\n".join(
        f"- {spec['name']}: {spec['label']}"
        + (
            f" ({spec['unit']})"
            if spec.get("unit")
            else ""
        )
        for spec in field_specs
    )

    return f"""Du analysierst den Text eines hochgeladenen
Geschäftsdokuments (z. B. Liefervertrag).

Für das erkannte Optimierungsproblem werden genau die folgenden
Parameter benötigt. Suche NUR nach diesen Werten im Dokumenttext,
keine anderen:

{field_lines}

Gib deine Antwort ausschließlich als JSON-Array zurück, ein Eintrag
pro oben genanntem Parameter, den du im Text findest (nicht gefundene
Parameter einfach weglassen). Verwende keine Markdown-Backticks und
keine zusätzlichen Erklärungen.

Format:
[
  {{
    "field_name": "selling_price",
    "label": "Verkaufspreis pro Einheit",
    "value": "4.50",
    "unit": "€",
    "confidence": 0.95,
    "evidence": "Der Preis pro Einheit beträgt 4,50 €."
  }}
]

Regeln:
- Erfinde keine Werte. Nimm ausschließlich Angaben auf, die wörtlich
  oder eindeutig im Text stehen.
- "field_name" muss exakt einem der oben genannten Namen entsprechen.
- "confidence" ist eine Zahl zwischen 0 und 1: wie sicher bist du dir,
  dass der Wert korrekt und vollständig aus dem Text gelesen wurde.
- "evidence" ist ein kurzes, nahezu wörtliches Zitat aus dem Text
  (maximal ein Satz), das den Wert belegt.
- Gib ein leeres Array zurück, wenn keiner der gesuchten Parameter im
  Text vorkommt.
"""


def extract_targeted_parameters(
    llm: BaseChatModel,
    document_text: str,
    field_specs: list[dict],
) -> list[ExtractedParameter]:
    """
    Wie `extract_document_parameters`, sucht aber gezielt nur nach
    den in `field_specs` übergebenen Parametern, statt nach
    beliebigen Werten. Das entspricht der „Targeted Extraction"
    innerhalb der Decision-first-Pipeline.
    """

    if not document_text.strip() or not field_specs:
        return []

    try:
        response = llm.invoke([
            SystemMessage(
                content=_build_targeted_extraction_prompt(
                    field_specs
                )
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


class DecisionFirstExtractionResult(TypedDict):
    """Ergebnis der zweistufigen Decision-first-Extraktion."""

    decision: str | None
    objective: str | None
    matched_family_name: str | None
    matched_variant_name: str | None
    parameters: list[ExtractedParameter]


def run_decision_first_extraction(
    llm: BaseChatModel,
    document_text: str,
) -> DecisionFirstExtractionResult:
    """
    Vereinfachte „Decision-first"-Extraktion in zwei Stufen:

    1. Erkennt grob, welche Entscheidung/welches Ziel im Dokument
       beschrieben ist (Wiederverwendung des Requirement-Agents).
    2. Passt eine registrierte Modellvariante zu dieser Einschätzung,
       wird NUR nach genau den Parametern gesucht, die diese Variante
       benötigt, statt nach beliebigen Werten.

    Passt keine Variante (z. B. weil das Dokument zu wenig Struktur
    hergibt), fällt die Funktion auf die allgemeine, ungerichtete
    Parametersuche zurück, statt leer zu bleiben.
    """

    requirements = identify_requirements_from_text(
        llm, document_text
    )

    match = find_matching_model_variant(requirements)

    if match is None:
        return {
            "decision": requirements.get("decision"),
            "objective": requirements.get("objective"),
            "matched_family_name": None,
            "matched_variant_name": None,
            "parameters": extract_document_parameters(
                llm, document_text
            ),
        }

    family_definition, variant_definition = match
    field_specs = build_field_specs_from_variant(
        variant_definition
    )

    return {
        "decision": requirements.get("decision"),
        "objective": requirements.get("objective"),
        "matched_family_name": family_definition[
            "family_name"
        ],
        "matched_variant_name": variant_definition[
            "variant_name"
        ],
        "parameters": extract_targeted_parameters(
            llm, document_text, field_specs
        ),
    }