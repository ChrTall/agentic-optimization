"""Gemeinsames Auslesen von JSON-Antworten der Sprachmodelle.

Alle Agenten erwarten vom Modell ein JSON-Objekt. Modelle liefern es aber
unterschiedlich verpackt: als reinen Text, in einem Markdown-Codeblock, mit
einleitendem Satz oder (bei Anthropic mit Thinking und der OpenAI Responses
API) als Liste von Inhaltsblöcken. Diese Funktionen gleichen das an einer
Stelle aus.
"""

import json
from typing import Any


def response_text(content: object) -> str:
    """Liefert den Textanteil einer Modellantwort.

    Unterstützt reinen Text und Listen von Inhaltsblöcken. Blöcke ohne Text
    (z. B. Thinking- oder Tool-Blöcke) werden übersprungen.
    """

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []

        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif (
                isinstance(block, dict)
                and block.get("type") in {None, "text", "output_text"}
                and isinstance(block.get("text"), str)
            ):
                parts.append(block["text"])

        if parts:
            return "".join(parts)

    raise TypeError("Die LLM-Antwort muss Text enthalten.")


def _strip_code_fence(text: str) -> str:
    lines = text.strip().splitlines()

    if (
        len(lines) >= 3
        and lines[0].strip().lower() in {"```", "```json"}
        and lines[-1].strip() == "```"
    ):
        return "\n".join(lines[1:-1]).strip()

    return text.strip()


def parse_json_object(
    content: object,
    *,
    not_object_message: str = (
        "Die LLM-Antwort muss ein JSON-Objekt enthalten."
    ),
) -> dict[str, Any]:
    """Dekodiert ein JSON-Objekt aus einer Modellantwort.

    Raises:
        TypeError: Die Antwort enthält keinen Text oder kein JSON-Objekt.
        ValueError: Die Antwort enthält kein gültiges JSON
            (``json.JSONDecodeError`` ist eine Unterklasse).
    """

    clean_content = _strip_code_fence(response_text(content))

    try:
        parsed = json.loads(clean_content)
    except json.JSONDecodeError:
        # Fallback: erstes JSON-Objekt im Text suchen, z. B. wenn das
        # Modell einen einleitenden Satz vor das JSON schreibt.
        start = clean_content.find("{")

        if start < 0:
            raise

        parsed, _ = json.JSONDecoder().raw_decode(clean_content[start:])

    if not isinstance(parsed, dict):
        raise TypeError(not_object_message)

    return parsed
