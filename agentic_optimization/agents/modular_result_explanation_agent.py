"""
Erzeugt eine verständliche Erklärung eines Optimierungsergebnisses.
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
Du erklärst das Ergebnis eines bereits gelösten
Optimierungsmodells für fachliche Anwender ohne vertiefte
Kenntnisse der mathematischen Optimierung.

Die Berechnung ist abgeschlossen. Du darfst weder das Modell
noch das Ergebnis verändern.

Erkläre in klarer deutscher Alltagssprache:

1. welche Entscheidung empfohlen wird,
2. was die wichtigsten Kennzahlen betrieblich bedeuten,
3. wie die Empfehlung zum Optimierungsziel passt,
4. welche ausdrücklich ausgeschlossenen Anforderungen nicht
   berücksichtigt wurden.

Wenn excluded_requirements leer ist, erwähne weder
ausgeschlossene Anforderungen noch die Tatsache, dass keine
solchen Anforderungen angegeben wurden. Unterscheide
Modellannahmen klar von ausdrücklich ausgeschlossenen
Nutzeranforderungen.

Verwende ausschließlich die übergebenen Modell- und
Ergebnisdaten. Ergänze keine neuen Werte, Ursachen,
Annahmen, Einsparungen oder Handlungsempfehlungen.

Interpretiere den Zielfunktionswert nur entsprechend der
übergebenen objective_summary. Bezeichne ihn nicht pauschal
als Gewinn, Kosten oder Umsatz.

Bei mehreren Produkten fasse das Ergebnis zusammen und
nenne nur besonders relevante Beispiele. Liste nicht alle
Tabellenwerte erneut auf.

Die Erklärung soll aus höchstens fünf kurzen Sätzen bestehen.
Verwende keine Formeln, keine Markdown-Überschriften, keine
Aufzählungen und keinen Programmcode.

Gib ausschließlich ein JSON-Objekt in diesem Format zurück:

{
  "explanation": "Verständlicher Erklärungstext"
}
""".strip()


ModularResultExplanationAgent = Callable[
    ...,
    str,
]


def _parse_explanation_response(
    content: object,
) -> str:
    """Prüft die strukturierte Antwort des Sprachmodells."""

    if not isinstance(content, str):
        raise TypeError(
            "Die Ergebnis-Erklärung muss als Text vorliegen."
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
            "Die Ergebnis-Erklärung muss als JSON-Objekt "
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
            "Ergebnis-Erklärung."
        )

    explanation = explanation.strip()

    if len(explanation) > 1800:
        raise ValueError(
            "Die erzeugte Ergebnis-Erklärung ist zu lang."
        )

    if "```" in explanation:
        raise ValueError(
            "Die Ergebnis-Erklärung darf keinen "
            "Programmcode enthalten."
        )

    return explanation


def create_modular_result_explanation_agent(
    llm: BaseChatModel,
) -> ModularResultExplanationAgent:
    """Erzeugt den Agenten für die Ergebnis-Erklärung."""

    def explain_result(
        *,
        review_data: dict[str, Any],
        result_data: dict[str, Any],
    ) -> str:
        payload = {
            "model": {
                "core_name": review_data["core_name"],
                "decision_description": review_data[
                    "decision_description"
                ],
                "objective_summary": review_data[
                    "objective_summary"
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
            },
            "result": {
                "status": result_data["status"],
                "status_label": result_data[
                    "status_label"
                ],
                "solution_method": result_data[
                    "solution_method_label"
                ],
                "objective_value": result_data[
                    "objective_value"
                ],
                "decisions": list(
                    result_data["decisions"]
                ),
                "metrics": list(
                    result_data["metrics"]
                ),
                "excluded_requirements": list(
                    result_data[
                        "excluded_requirements"
                    ]
                ),
            },
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

    return explain_result