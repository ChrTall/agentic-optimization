"""
Erzeugt eine verständliche Erklärung für die Modellprüfung.
"""

import json
from collections.abc import Callable
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)


SYSTEM_PROMPT = """
Du erklärst ein bereits vollständig zusammengestelltes
Optimierungsmodell für fachliche Anwender ohne vertiefte
Kenntnisse der mathematischen Optimierung.
Verwende gespeicherte display_name bzw. product_name statt interner
product_id. Erfinde keine Produktnamen.

Die Modellwahl, Zielfunktion, Nebenbedingungen und Daten
wurden bereits festgelegt. Du darfst daran nichts verändern.

Erkläre in klarer deutscher Alltagssprache:

1. unter welchen belegten Voraussetzungen das ausgewählte
   Grundmodell zur Situation passt,
2. welche Entscheidung berechnet wird,
3. welches wirtschaftliche Ziel verfolgt wird,
4. welche zusätzlichen betrieblichen Vorgaben berücksichtigt
   werden.

Verwende ausschließlich Informationen aus den übergebenen
Review-Daten. Ergänze keine neuen Annahmen, Werte,
Nebenbedingungen oder fachlichen Behauptungen.

Wenn Anforderungen ausdrücklich ausgeschlossen wurden,
weise knapp darauf hin, dass sie nicht Teil der Berechnung
sind.

Falls "pending_core_assumptions" nicht leer ist, sind diese
Modellvoraussetzungen noch nicht bestätigt. Beschreibe die
Modellwahl dann ausdrücklich als vorläufig und behaupte nicht,
dass alle Voraussetzungen bereits erfüllt sind. Gib die offenen
Voraussetzungen verständlich wieder.

Die Erklärung soll aus höchstens vier kurzen Sätzen bestehen.
Verwende keine Formeln, keine Markdown-Überschriften, keine
Aufzählungen und keinen Programmcode.

Gib ausschließlich ein JSON-Objekt in diesem Format zurück:

{
  "explanation": "Verständlicher Erklärungstext"
}
""".strip()


ModularReviewExplanationAgent = Callable[
    ...,
    str,
]


def _parse_explanation_response(
    content: object,
) -> str:
    """Prüft die strukturierte Antwort des Sprachmodells."""

    if not isinstance(content, str):
        raise TypeError(
            "Die Erklärung des Sprachmodells muss "
            "als Text vorliegen."
        )

    clean_content = content.strip()
    lines = clean_content.splitlines()

    if (
        len(lines) >= 3
        and lines[0].strip().lower()
        in {
            "```",
            "```json",
        }
        and lines[-1].strip() == "```"
    ):
        clean_content = "\n".join(
            lines[1:-1]
        ).strip()

    parsed_response = json.loads(clean_content)

    if not isinstance(parsed_response, dict):
        raise TypeError(
            "Die Erklärung muss als JSON-Objekt "
            "zurückgegeben werden."
        )

    if set(parsed_response) != {"explanation"}:
        raise ValueError(
            "Die LLM-Antwort darf ausschließlich das Feld "
            "'explanation' enthalten."
        )

    explanation = parsed_response["explanation"]

    if (
        not isinstance(explanation, str)
        or not explanation.strip()
    ):
        raise ValueError(
            "Die LLM-Antwort enthält keine gültige "
            "Erklärung."
        )

    explanation = explanation.strip()

    if len(explanation) > 1500:
        raise ValueError(
            "Die erzeugte Erklärung ist zu lang."
        )

    if "```" in explanation:
        raise ValueError(
            "Die erzeugte Erklärung darf keinen "
            "Programmcode enthalten."
        )

    return explanation


def create_modular_review_explanation_agent(
    llm: BaseChatModel,
) -> ModularReviewExplanationAgent:
    """Erzeugt den Agenten für die Review-Erklärung."""

    def explain_review(
        *,
        review_data: dict[str, Any],
    ) -> str:
        payload = {
            "core_name": review_data["core_name"],
            "core_description": review_data[
                "core_description"
            ],
            "decision_description": review_data[
                "decision_description"
            ],
            "product_count": len(
                review_data["products"]
            ),
            "objective_summary": review_data[
                "objective_summary"
            ],
            "objective_components": [
                component["name"]
                for component in review_data[
                    "objective_components"
                ]
            ],
            "constraints": [
                {
                    "name": constraint["name"],
                    "product_id": constraint[
                        "product_id"
                    ],
                    "parameters": constraint[
                        "parameters"
                    ],
                }
                for constraint in review_data[
                    "constraints"
                ]
            ],
            "assumptions": list(
                review_data["assumptions"]
            ),
            "pending_core_assumptions": list(
                review_data.get(
                    "pending_core_assumptions", []
                )
            ),
            "excluded_requirements": [
                {
                    "description": requirement[
                        "description"
                    ],
                    "reason": requirement["reason"],
                }
                for requirement in review_data[
                    "excluded_requirements"
                ]
            ],
        }

        response = llm.invoke(
            [
                SystemMessage(
                    content=SYSTEM_PROMPT
                ),
                HumanMessage(
                    content=json.dumps(
                        payload,
                        ensure_ascii=False,
                        indent=2,
                    )
                ),
            ]
        )

        return _parse_explanation_response(
            response.content
        )

    return explain_review
