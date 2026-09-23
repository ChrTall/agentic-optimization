import json
from unittest.mock import Mock

from langchain_core.messages import AIMessage

from agentic_optimization.agents.modular_workflow_nodes import (
    create_modular_result_explanation_node,
)
from agentic_optimization.models.proposal_mapper import (
    build_specification_from_proposal,
)
from agentic_optimization.state import (
    WORKFLOW_OPTIMIZATION_COMPLETED,
    get_workflow_status,
)


def create_state() -> dict:
    proposal = {
        "core_id": "eoq",
        "selection_status": "complete",
        "objective_sense": "minimize",
        "objective_components": [
            {
                "component_id": "annual_ordering_cost",
                "reason": "Bestellkosten berücksichtigen.",
                "sources": ["raw_input"],
            },
            {
                "component_id": "annual_holding_cost",
                "reason": "Lagerkosten berücksichtigen.",
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
                "expected_value": 3500.0,
                "horizon_days": 365,
            },
            "facts": {
                "ordering_cost": 120.0,
                "holding_cost": 12.0,
            },
        }
    ]
    specification = build_specification_from_proposal(
        proposal,
        products,
        {},
    )
    optimization_result = {
        "status": "optimal",
        "solution_method": "analytical",
        "objective_value": 3174.9,
        "decisions": [
            {
                "quantity_id": "order_quantity",
                "product_id": "product_1",
                "value": 264.58,
            }
        ],
        "metrics": [
            {
                "metric_id": "orders_per_year",
                "product_id": "product_1",
                "value": 13.23,
            }
        ],
        "technical_representation": None,
        "errors": [],
    }

    return {
        "control": {
            "workflow_status": (
                WORKFLOW_OPTIMIZATION_COMPLETED
            ),
        },
        "modular_model": {
            "core_id": "eoq",
            "products": products,
            "global_facts": {},
            "proposal": proposal,
            "specification": specification,
            "optimization_result": optimization_result,
        },
    }


def test_stores_llm_result_explanation() -> None:
    llm = Mock()
    llm.invoke.return_value = AIMessage(
        content=json.dumps(
            {
                "explanation": (
                    "Empfohlen wird eine Bestellmenge von "
                    "264,58 Einheiten."
                )
            }
        )
    )
    node = create_modular_result_explanation_node(
        llm
    )

    result = node(create_state())

    assert result["modular_model"][
        "result_explanation"
    ].startswith("Empfohlen wird")
    assert result["modular_model"][
        "result_explanation_source"
    ] == "llm"
    assert get_workflow_status(result) == (
        WORKFLOW_OPTIMIZATION_COMPLETED
    )
    assert llm.invoke.call_count == 1


def test_uses_result_fallback_for_invalid_response() -> None:
    llm = Mock()
    llm.invoke.return_value = AIMessage(
        content="ungültige Antwort"
    )
    node = create_modular_result_explanation_node(
        llm
    )

    result = node(create_state())

    assert result["modular_model"][
        "result_explanation_source"
    ] == "fallback"
    assert result["modular_model"][
        "result_explanation"
    ].startswith("Optimale Lösung gefunden")
    assert get_workflow_status(result) == (
        WORKFLOW_OPTIMIZATION_COMPLETED
    )


def test_reuses_stored_result_explanation() -> None:
    llm = Mock()
    state = create_state()
    state["modular_model"]["result_explanation"] = (
        "Bereits erzeugte Ergebnis-Erklärung."
    )
    state["modular_model"][
        "result_explanation_source"
    ] = "llm"

    node = create_modular_result_explanation_node(
        llm
    )
    result = node(state)

    assert result["modular_model"][
        "result_explanation"
    ] == "Bereits erzeugte Ergebnis-Erklärung."
    assert llm.invoke.call_count == 0
    assert get_workflow_status(result) == (
        WORKFLOW_OPTIMIZATION_COMPLETED
    )
