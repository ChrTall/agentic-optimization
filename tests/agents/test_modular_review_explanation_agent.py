import json
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage

from agentic_optimization.agents.modular_review_explanation_agent import (
    create_modular_review_explanation_agent,
)


def create_review_data() -> dict:
    return {
        "core_id": "single_period_newsvendor",
        "core_name": "Einperiodiges Newsvendor-Modell",
        "core_description": (
            "Das Modell unterstützt eine einmalige "
            "Bestellentscheidung bei unsicherer Nachfrage."
        ),
        "decision_description": (
            "Berechnet wird die Bestellmenge."
        ),
        "products": [
            {
                "product_id": "product_1",
                "forecast": {},
                "facts": {},
            }
        ],
        "assumptions": [
            "Während des Zeitraums wird nicht nachbestellt."
        ],
        "excluded_requirements": [],
        "global_facts": {},
        "objective_sense": "maximize",
        "objective_summary": (
            "Verkaufserlös − Einkaufskosten maximieren"
        ),
        "objective_components": [
            {
                "component_id": "sales_revenue",
                "name": "Verkaufserlös",
                "effect": "benefit",
                "reason": "Verkaufserlöse berücksichtigen.",
                "sources": ["raw_input"],
            },
            {
                "component_id": "purchase_cost",
                "name": "Einkaufskosten",
                "effect": "cost",
                "reason": "Einkaufskosten berücksichtigen.",
                "sources": ["raw_input"],
            },
        ],
        "constraints": [
            {
                "component_id": "minimum_order_quantity",
                "name": "Mindestbestellmenge",
                "scope": "per_product",
                "product_id": "product_1",
                "parameters": {
                    "quantity": 50.0,
                },
                "reason": "Mindestmenge wurde genannt.",
                "sources": ["raw_input"],
            }
        ],
        "missing_inputs": [],
        "validation_errors": [],
        "ready_for_confirmation": True,
    }


def test_generates_review_explanation() -> None:
    llm = Mock()
    llm.invoke.return_value = AIMessage(
        content=json.dumps(
            {
                "explanation": (
                    "Für die einmalige Bestellung wird das "
                    "Newsvendor-Modell verwendet. Es bestimmt "
                    "die passende Bestellmenge und maximiert "
                    "Verkaufserlöse abzüglich Einkaufskosten. "
                    "Dabei gilt eine Mindestbestellmenge."
                )
            }
        )
    )

    agent = create_modular_review_explanation_agent(
        llm
    )

    explanation = agent(
        review_data=create_review_data()
    )

    assert explanation.startswith(
        "Für die einmalige Bestellung"
    )

    messages = llm.invoke.call_args.args[0]

    assert (
        "Ergänze keine neuen Annahmen"
        in messages[0].content
    )

    payload = json.loads(messages[1].content)

    assert payload["product_count"] == 1
    assert payload["objective_summary"] == (
        "Verkaufserlös − Einkaufskosten maximieren"
    )
    assert payload["constraints"] == [
        {
            "name": "Mindestbestellmenge",
            "product_id": "product_1",
            "parameters": {
                "quantity": 50.0,
            },
        }
    ]


def test_rejects_empty_review_explanation() -> None:
    llm = Mock()
    llm.invoke.return_value = AIMessage(
        content=json.dumps(
            {
                "explanation": "   ",
            }
        )
    )

    agent = create_modular_review_explanation_agent(
        llm
    )

    with pytest.raises(
        ValueError,
        match="keine gültige Erklärung",
    ):
        agent(
            review_data=create_review_data()
        )


def test_passes_pending_assumptions_to_explanation_llm() -> None:
    llm = Mock()
    llm.invoke.return_value = AIMessage(
        content=json.dumps({
            "explanation": "Die Modellwahl ist vorläufig."
        })
    )
    review_data = create_review_data()
    review_data["pending_core_assumptions"] = [
        "Der Bedarf ist ungefähr gleichmäßig."
    ]

    create_modular_review_explanation_agent(llm)(
        review_data=review_data
    )

    messages = llm.invoke.call_args.args[0]
    payload = json.loads(messages[1].content)
    assert payload["pending_core_assumptions"] == [
        "Der Bedarf ist ungefähr gleichmäßig."
    ]
    assert "Modellwahl dann ausdrücklich als vorläufig" in (
        messages[0].content
    )
