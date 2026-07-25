import json
import re
from collections.abc import Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from agentic_optimization.state import (
    WORKFLOW_GENERAL_REQUIREMENTS_EXTRACTED,
    GraphState,
    with_control,
)

REQUIRED_FIELDS = ["decision", "objective", "demand_type"]

SYSTEM_PROMPT = """Du bist ein Assistent, der Optimierungsprobleme analysiert.

Der Nutzer hat ein Formular ausgefüllt und optional einen Freitext eingegeben.
Deine Aufgabe: Extrahiere die vier wichtigsten Angaben für ein Optimierungsmodell.

Gib deine Antwort NUR als JSON zurück, ohne Erklärungen, ohne Markdown-Backticks.

Format:
{
  "decision": "...",
  "objective": "...",
  "demand_type": "...",
  "constraints": ["...", "..."],
  "missing_fields": ["field1", "field2"]
}

Felder:
- decision: Welche Entscheidung soll optimiert werden? (z.B. Bestellmenge, Produktionsmenge)
- objective: Was ist das Ziel? (z.B. Gewinn maximieren, Kosten minimieren)
- demand_type: Wie ist die Nachfrage? (z.B. sicher, unsicher, saisonal)
- constraints: Welche Nebenbedingungen gibt es? (z.B. Budget, Kapazität, Lagerplatz)

Wenn eine Information weder im Formular noch im Freitext erkennbar ist, setze das Feld auf null
und füge den Feldnamen in missing_fields ein.
"""


def create_requirement_agent(llm: BaseChatModel) -> Callable[[GraphState], GraphState]:
    def requirement_agent(state: GraphState) -> GraphState:
        form_data = state.get("form_data", {})
        raw_input = state.get("raw_input", "")

        user_message = _build_user_message(form_data, raw_input)

        print("=== LLM INPUT ===")
        print(user_message)

        try:
            extracted = _extract_with_llm(llm, user_message)
            print("=== LLM OUTPUT ===")
            print(extracted)
        except Exception as e:
            print(f"=== LLM FEHLER: {e} ===")
            extracted = _extract_rule_based(form_data, raw_input)

        constraints = extracted.get("constraints") or []

        if isinstance(constraints, str):
            constraints = [constraints]

        if not isinstance(constraints, list):
            constraints = []

        missing_fields = extracted.get("missing_fields") or [
            field
            for field in REQUIRED_FIELDS
            if not extracted.get(field)
        ]

        if isinstance(missing_fields, str):
            missing_fields = [missing_fields]

        if not isinstance(missing_fields, list):
            missing_fields = [
                field
                for field in REQUIRED_FIELDS
                if not extracted.get(field)
            ]

        missing_fields = [
            field
            for field in missing_fields
            if field in REQUIRED_FIELDS
        ]

        requirements = {
            "decision": extracted.get("decision"),
            "objective": extracted.get("objective"),
            "demand_type": extracted.get("demand_type"),
            "constraints": constraints,
            "raw_description": raw_input,
        }

        return with_control(
            {
                **state,
                "requirements": requirements,
                "missing_fields": missing_fields,
                "clarification_question": "",
                "general_validation_errors": [],
                "consistency_warnings": [],
            },
            event=None,
            workflow_status=WORKFLOW_GENERAL_REQUIREMENTS_EXTRACTED,
        )

    return requirement_agent


# ── LLM ────────────────────────────────────────────────────────────────────

def _extract_with_llm(llm: BaseChatModel, user_message: str) -> dict:
    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ])

    raw_text = str(response.content)
    return _parse_json(raw_text)


# ── Hilfsfunktionen ───────────────────────────────────────────────────────────

def _build_user_message(form_data: dict, raw_input: str) -> str:
    lines = ["Formulareingaben:"]
    for key, value in form_data.items():
        lines.append(f"  {key}: {value}")
    if raw_input:
        lines.append(f"\nFreitext des Nutzers:\n{raw_input}")
    return "\n".join(lines)


def _parse_json(text: str) -> dict:
    """Extrahiert JSON aus LLM-Antwort, auch wenn Markdown-Backticks vorhanden."""
    clean = re.sub(r"```(?:json)?|```", "", text).strip()
    parsed = json.loads(clean)

    if not isinstance(parsed, dict):
        raise TypeError(
            "Die LLM-Antwort muss ein JSON-Objekt sein."
        )

    return parsed


def _extract_rule_based(form_data: dict, raw_input: str) -> dict:
    """Fallback: direkte Übernahme der Formularfelder."""
    return {
        "decision": form_data.get("decision"),
        "objective": form_data.get("objective"),
        "demand_type": form_data.get("demand_type"),
        "constraints": form_data.get("constraints"),
        "missing_fields": [
            f for f in REQUIRED_FIELDS
            if not form_data.get(f)
        ],
    }