import json
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage

from agentic_optimization.agents.modular_result_explanation_agent import (
    create_modular_result_explanation_agent,
)


def create_review_data() -> dict:
    return {
        "core_name": "Wirtschaftliche Bestellmenge (EOQ)",
        "decision_description": (
            "Berechnet wird die Bestellmenge je Bestellung."
        ),
        "objective_summary": (
            "Jährliche Bestellkosten + jährliche "
            "Lagerhaltungskosten minimieren"
        ),
        "constraints": [],
        "assumptions": [
            "Der Bedarf wird als gleichmäßig angenommen."
        ],
    }


def create_result_data() -> dict:
    return {
        "status": "optimal",
        "status_label": "Optimale Lösung gefunden",
        "solution_method_label": (
            "Analytische Berechnung"
        ),
        "objective_value": 3174.9,
        "decisions": [
            {
                "value_id": "order_quantity",
                "label": "Bestellmenge",
                "value": 264.58,
                "product_id": "product_1",
                "aggregation": None,
            }
        ],
        "metrics": [
            {
                "value_id": "orders_per_year",
                "label": "Bestellungen pro Jahr",
                "value": 13.23,
                "product_id": "product_1",
                "aggregation": None,
            }
        ],
        "excluded_requirements": [],
    }


def test_generates_result_explanation() -> None:
    llm = Mock()
    llm.invoke.return_value = AIMessage(
        content=json.dumps(
            {
                "explanation": (
                    "Empfohlen wird eine Bestellmenge von "
                    "264,58 Einheiten. Daraus ergeben sich "
                    "etwa 13,23 Bestellungen pro Jahr."
                )
            }
        )
    )

    agent = create_modular_result_explanation_agent(
        llm
    )
    explanation = agent(
        review_data=create_review_data(),
        result_data=create_result_data(),
    )

    assert explanation.startswith(
        "Empfohlen wird eine Bestellmenge"
    )

    messages = llm.invoke.call_args.args[0]

    assert (
        "Ergänze keine neuen Werte"
        in messages[0].content
    )

    assert (
        "Wenn excluded_requirements leer ist"
        in messages[0].content
    )

    payload = json.loads(messages[1].content)

    assert payload["model"]["objective_summary"] == (
        "Jährliche Bestellkosten + jährliche "
        "Lagerhaltungskosten minimieren"
    )
    assert payload["result"]["decisions"][0][
        "value"
    ] == 264.58


def test_rejects_additional_response_fields() -> None:
    llm = Mock()
    llm.invoke.return_value = AIMessage(
        content=json.dumps(
            {
                "explanation": "Eine Erklärung.",
                "recommendation": "Mehr bestellen.",
            }
        )
    )

    agent = create_modular_result_explanation_agent(
        llm
    )

    with pytest.raises(
        ValueError,
        match="ausschließlich das Feld",
    ):
        agent(
            review_data=create_review_data(),
            result_data=create_result_data(),
        )
