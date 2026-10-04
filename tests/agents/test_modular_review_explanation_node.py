import json
from unittest.mock import Mock

from langchain_core.messages import AIMessage

from agentic_optimization.agents.modular_workflow_nodes import (
    create_modular_review_explanation_node,
)
from agentic_optimization.models.core_applicability import (
    CONDITION_GUIDANCE,
)
from agentic_optimization.models.proposal_mapper import (
    build_specification_from_proposal,
)
from agentic_optimization.state import (
    WORKFLOW_READY_FOR_REVIEW,
    get_workflow_status,
)


def create_state() -> dict:
    proposal = {
        "core_id": "single_period_newsvendor",
        "selection_status": "complete",
        "objective_sense": "maximize",
        "objective_components": [
            {
                "component_id": "sales_revenue",
                "reason": "Verkaufserlöse berücksichtigen.",
                "sources": ["raw_input"],
            },
            {
                "component_id": "purchase_cost",
                "reason": "Einkaufskosten berücksichtigen.",
                "sources": ["raw_input"],
            },
        ],
        "constraints": [],
        "unresolved_requirements": [],
    }
    products = [
        {
            "product_id": "product_1",
            "forecast": {
                "expected_value": 100.0,
                "quantiles": [
                    {
                        "level": 0.5,
                        "value": 100.0,
                    }
                ],
                "horizon_days": 30,
            },
            "facts": {
                "selling_price": 12.0,
                "purchase_cost": 5.0,
            },
        }
    ]
    specification = build_specification_from_proposal(
        proposal,
        products,
        {},
    )

    return {
        "modular_model": {
            "core_id": "single_period_newsvendor",
            "products": products,
            "global_facts": {},
            "proposal": proposal,
            "specification": specification,
        }
    }


def test_stores_llm_review_explanation() -> None:
    llm = Mock()
    llm.invoke.return_value = AIMessage(
        content=json.dumps(
            {
                "explanation": (
                    "Das Newsvendor-Modell bestimmt eine "
                    "einmalige Bestellmenge bei unsicherer "
                    "Nachfrage."
                )
            }
        )
    )
    node = create_modular_review_explanation_node(
        llm
    )

    result = node(create_state())

    assert result["modular_model"][
        "review_explanation"
    ].startswith("Das Newsvendor-Modell")
    assert result["modular_model"][
        "review_explanation_source"
    ] == "llm"
    assert get_workflow_status(result) == (
        WORKFLOW_READY_FOR_REVIEW
    )
    assert llm.invoke.call_count == 1


def test_uses_fallback_for_invalid_llm_response() -> None:
    llm = Mock()
    llm.invoke.return_value = AIMessage(
        content="keine gültige JSON-Antwort"
    )
    node = create_modular_review_explanation_node(
        llm
    )

    result = node(create_state())

    assert result["modular_model"][
        "review_explanation_source"
    ] == "fallback"
    assert result["modular_model"][
        "review_explanation"
    ].startswith(
        "Für ein Produkt wird das Modell"
    )
    assert get_workflow_status(result) == (
        WORKFLOW_READY_FOR_REVIEW
    )


def test_reuses_stored_review_explanation() -> None:
    llm = Mock()
    state = create_state()
    state["modular_model"]["review_explanation"] = (
        "Bereits erzeugte Erklärung."
    )
    state["modular_model"][
        "review_explanation_source"
    ] = "llm"

    node = create_modular_review_explanation_node(
        llm
    )
    result = node(state)

    assert result["modular_model"][
        "review_explanation"
    ] == "Bereits erzeugte Erklärung."
    assert llm.invoke.call_count == 0
    assert get_workflow_status(result) == (
        WORKFLOW_READY_FOR_REVIEW
    )


def test_fallback_marks_unconfirmed_model_choice_as_provisional() -> None:
    llm = Mock()
    llm.invoke.return_value = AIMessage(content="kein JSON")
    state = create_state()
    state["modular_model"]["pending_core_assumptions"] = [
        "no_replenishment_within_period"
    ]

    result = create_modular_review_explanation_node(llm)(
        state
    )

    explanation = result["modular_model"][
        "review_explanation"
    ]
    assert "Diese Modellwahl ist vorläufig" in explanation
    assert CONDITION_GUIDANCE[
        "no_replenishment_within_period"
    ] in explanation
