import json
from unittest.mock import Mock

from langchain_core.messages import AIMessage

import pytest
from agentic_optimization.agents.modular_workflow_nodes import (
    apply_modular_model_inputs_node,
    confirm_modular_requirement_exclusions_node,
    create_modular_model_preparation_node,
    execute_modular_model_node,
    create_modular_product_schema_node,
)
from agentic_optimization.state import (
    WORKFLOW_NEEDS_MODEL_SPECIFIC_INPUTS,
    WORKFLOW_READY_FOR_REVIEW,
    WORKFLOW_OPTIMIZATION_COMPLETED,
    WORKFLOW_MODEL_UNSUPPORTED,
    WORKFLOW_NEEDS_GENERAL_CLARIFICATION,
    WORKFLOW_MODEL_PREPARATION_FAILED,
    WORKFLOW_PRODUCT_TABLE_READY,
    get_workflow_status,
)
from agentic_optimization.agents import (
    modular_workflow_nodes,
)


def test_prepares_modular_model_in_graph_state() -> None:
    response = {
        "core_id": "continuous_review_q_r",
        "selection_status": "complete",
        "objective_sense": "minimize",
        "objective_components": [
            {
                "component_id": "annual_ordering_cost",
                "reason": "Bestellkosten wurden genannt.",
                "sources": ["source_1"],
            },
            {
                "component_id": "annual_holding_cost",
                "reason": "Lagerkosten wurden genannt.",
                "sources": ["source_1"],
            },
            {
                "component_id": "annual_shortage_cost",
                "reason": "Fehlmengenkosten wurden genannt.",
                "sources": ["source_1"],
            },
        ],
        "constraints": [],
        "unresolved_requirements": [],
    }

    llm = Mock()
    llm.invoke.side_effect = [
        AIMessage(
            content=json.dumps(
                {
                    "products": [],
                    "global_facts": {},
                }
            )
        ),
        AIMessage(
            content=json.dumps(response)
        ),
    ]

    node = create_modular_model_preparation_node(llm)

    state = {
        "raw_input": "Bestand kontinuierlich überwachen.",
        "requirements": {
            "objective": "minimize_cost",
        },
        "modular_model": {
            "core_id": "continuous_review_q_r",
            "products": [
                {
                    "product_id": "product_1",
                    "forecast": {
                        "expected_value": 100.0,
                        "quantiles": [
                            {
                                "level": 0.1,
                                "value": 70.0,
                            },
                            {
                                "level": 0.5,
                                "value": 100.0,
                            },
                            {
                                "level": 0.9,
                                "value": 140.0,
                            },
                        ],
                        "horizon_days": 30,
                    },
                    "facts": {
                        "lead_time_days": 10,
                    },
                }
            ],
            "global_facts": {},
            "evidence": [
                {
                    "source_id": "source_1",
                    "content": (
                        "Bestellkosten, Lagerkosten und "
                        "Fehlmengenkosten berücksichtigen."
                    ),
                }
            ],
            "preliminary_proposal": response,
        },
    }

    result = node(state)

    modular_result = result["modular_model"]

    assert modular_result["proposal"] == response
    assert modular_result["specification"]["core_id"] == (
        "continuous_review_q_r"
    )
    assert modular_result["validation"]["is_valid"] is False
    assert {
        missing_input["field"]["name"]
        for missing_input
        in modular_result["validation"]["missing_inputs"]
    } == {
        "ordering_cost",
        "holding_cost",
        "shortage_cost",
    }

    assert get_workflow_status(result) == (
        WORKFLOW_NEEDS_MODEL_SPECIFIC_INPUTS
    )
    assert result["raw_input"] == state["raw_input"]
    assert modular_result["review_confirmed"] is False
    assert modular_result["optimization_result"] is None

    assert llm.invoke.call_count == 2

    values = {
        "ordering_cost": 50.0,
        "holding_cost": 2.0,
        "shortage_cost": 20.0,
    }

    modular_result["pending_input_submissions"] = [
        {
            "missing_input": missing_input,
            "value": values[
                missing_input["field"]["name"]
            ],
        }
        for missing_input
        in modular_result["validation"]["missing_inputs"]
    ]

    completed_state = apply_modular_model_inputs_node(
        result
    )
    completed_model = completed_state["modular_model"]

    assert completed_model["validation"]["is_valid"] is True
    assert completed_model["validation"]["errors"] == []
    assert completed_model["validation"][
        "missing_inputs"
    ] == []
    assert completed_model[
        "pending_input_submissions"
    ] == []
    assert completed_model["review_confirmed"] is False
    assert completed_model["optimization_result"] is None

    assert get_workflow_status(completed_state) == (
        WORKFLOW_READY_FOR_REVIEW
    )

    product_facts = completed_model["specification"][
        "products"
    ][0]["facts"]

    assert product_facts["ordering_cost"] == 50.0
    assert product_facts["holding_cost"] == 2.0
    assert product_facts["shortage_cost"] == 20.0

    assert llm.invoke.call_count == 2

    with pytest.raises(
        ValueError,
        match="bestätigt",
    ):
        execute_modular_model_node(completed_state)

    confirmed_state = {
        **completed_state,
        "modular_model": {
            **completed_model,
            "review_confirmed": True,
        },
    }

    optimized_state = execute_modular_model_node(
        confirmed_state
    )
    optimized_model = optimized_state["modular_model"]
    optimization_result = optimized_model[
        "optimization_result"
    ]

    assert optimization_result is not None
    assert optimization_result["status"] == "optimal"
    assert optimization_result["solution_method"] == (
        "numerical"
    )
    assert {
        decision["quantity_id"]
        for decision in optimization_result["decisions"]
    } == {
        "order_quantity",
        "reorder_point",
    }

    assert get_workflow_status(optimized_state) == (
        WORKFLOW_OPTIMIZATION_COMPLETED
    )

    assert llm.invoke.call_count == 2


@pytest.mark.parametrize(
    (
        "selection_status",
        "resolution",
        "expected_workflow_status",
    ),
    [
        (
            "needs_clarification",
            "clarification",
            WORKFLOW_NEEDS_GENERAL_CLARIFICATION,
        ),
        (
            "unsupported",
            "unsupported",
            WORKFLOW_MODEL_UNSUPPORTED,
        ),
    ],
)
def test_routes_unresolved_component_selection(
    monkeypatch: pytest.MonkeyPatch,
    selection_status: str,
    resolution: str,
    expected_workflow_status: str,
) -> None:
    proposal = {
        "core_id": "eoq",
        "selection_status": selection_status,
        "objective_sense": "minimize",
        "objective_components": [
            {
                "component_id": "annual_ordering_cost",
                "reason": "Bestellkosten wurden genannt.",
                "sources": ["source_1"],
            },
            {
                "component_id": "annual_holding_cost",
                "reason": "Lagerkosten wurden genannt.",
                "sources": ["source_1"],
            },
        ],
        "constraints": [],
        "unresolved_requirements": [
            {
                "description": "Mengenrabatt",
                "resolution": resolution,
                "candidate_component_ids": (
                    [
                        "minimum_order_quantity",
                        "maximum_order_quantity",
                    ]
                    if resolution == "clarification"
                    else []
                ),
                "reason": (
                    "Die Anforderung kann derzeit nicht "
                    "vollständig abgebildet werden."
                ),
                "sources": ["source_1"],
            }
        ],
    }

    def prepare_model(**kwargs):
        return {
            "proposal": proposal,
            "products": kwargs["products"],
            "global_facts": kwargs["global_facts"],
        }
    monkeypatch.setattr(
        modular_workflow_nodes,
        "create_modular_model_agent",
        lambda _llm, **_kwargs: prepare_model,
    )

    node = (
        modular_workflow_nodes
        .create_modular_model_preparation_node(Mock())
    )

    result = node(
        {
            "requirements": {},
            "modular_model": {
                "core_id": "eoq",
                "products": [],
                "global_facts": {},
                "evidence": [],
            },
        }
    )

    modular_state = result["modular_model"]

    assert modular_state["proposal"] == proposal
    assert modular_state["products"] == []
    assert modular_state["global_facts"] == {}
    assert "specification" not in modular_state
    assert "validation" not in modular_state
    assert get_workflow_status(result) == (
        expected_workflow_status
    )


def test_handles_failed_model_preparation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def prepare_model(**_kwargs):
        raise ValueError(
            "Ungültige LLM-Antwort"
        )

    monkeypatch.setattr(
        modular_workflow_nodes,
        "create_modular_model_agent",
        lambda _llm, **_kwargs: prepare_model,
    )

    node = (
        modular_workflow_nodes
        .create_modular_model_preparation_node(Mock())
    )

    result = node(
        {
            "requirements": {},
            "modular_model": {
                "core_id": "eoq",
                "products": [],
                "global_facts": {},
                "evidence": [],
            },
        }
    )

    modular_state = result["modular_model"]

    assert get_workflow_status(result) == (
        WORKFLOW_MODEL_PREPARATION_FAILED
    )
    assert modular_state["preparation_error"] == (
        "Die automatische Modellzusammenstellung "
        "konnte nicht zuverlässig abgeschlossen "
        "werden. Bitte starten Sie die Analyse erneut."
    )
    assert "proposal" not in modular_state
    assert "specification" not in modular_state
    assert "validation" not in modular_state


def test_continues_after_confirmed_requirement_exclusion(
) -> None:
    proposal = {
        "core_id": "eoq",
        "selection_status": "unsupported",
        "objective_sense": "minimize",
        "objective_components": [
            {
                "component_id": "annual_ordering_cost",
                "reason": "Bestellkosten wurden genannt.",
                "sources": ["raw_input"],
            },
            {
                "component_id": "annual_holding_cost",
                "reason": "Lagerkosten wurden genannt.",
                "sources": ["raw_input"],
            },
        ],
        "constraints": [],
        "unresolved_requirements": [
            {
                "description": (
                    "Die Lieferung muss an einem "
                    "Dienstag erfolgen."
                ),
                "resolution": "unsupported",
                "candidate_component_ids": [],
                "reason": (
                    "Für feste Liefertage ist kein "
                    "kompatibler Baustein verfügbar."
                ),
                "sources": ["raw_input"],
            }
        ],
    }

    state = {
        "modular_model": {
            "core_id": "eoq",
            "products": [
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
            ],
            "global_facts": {},
            "proposal": proposal,
            "excluded_requirements": [],
            "review_confirmed": False,
            "optimization_result": None,
        },
    }

    result = (
        confirm_modular_requirement_exclusions_node(
            state
        )
    )

    modular_result = result["modular_model"]

    assert modular_result["proposal"][
        "selection_status"
    ] == "complete"
    assert modular_result["proposal"][
        "unresolved_requirements"
    ] == []

    assert modular_result[
        "excluded_requirements"
    ][0]["description"] == (
        "Die Lieferung muss an einem Dienstag erfolgen."
    )

    assert modular_result["specification"][
        "excluded_requirements"
    ] == modular_result["excluded_requirements"]

    assert modular_result["validation"]["is_valid"] is True
    assert get_workflow_status(result) == (
        WORKFLOW_READY_FOR_REVIEW
    )

    assert proposal["selection_status"] == "unsupported"
    assert len(proposal["unresolved_requirements"]) == 1


def test_prepares_product_table_schema_before_products_exist(
) -> None:
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

    llm = Mock()
    llm.invoke.return_value = AIMessage(
        content=json.dumps(proposal)
    )

    node = create_modular_product_schema_node(llm)

    result = node(
        {
            "requirements": {
                "objective": "maximize_profit",
            },
            "modular_model": {
                "core_id": "single_period_newsvendor",
                "products": [],
                "global_facts": {},
                "evidence": [
                    {
                        "source_id": "raw_input",
                        "content": (
                            "Der erwartete Gewinn soll "
                            "maximiert werden."
                        ),
                    }
                ],
            },
        }
    )

    assert get_workflow_status(result) == (
        WORKFLOW_PRODUCT_TABLE_READY
    )
    preliminary = result["modular_model"]["preliminary_proposal"]
    assert [
        item["component_id"] for item in preliminary["objective_components"]
    ] == [
        "sales_revenue", "purchase_cost"
    ]
    assert preliminary["constraints"] == []
    llm.invoke.assert_not_called()
    assert result["modular_model"][
        "products"
    ] == []


def test_corrects_existing_invalid_product_fact() -> None:
    specification = {
        "core_id": "single_period_newsvendor",
        "products": [
            {
                "product_id": "product_1",
                "forecast": {
                    "expected_value": 100.0,
                    "quantiles": [
                        {"level": 0.1, "value": 70.0},
                        {"level": 0.5, "value": 100.0},
                        {"level": 0.9, "value": 140.0},
                    ],
                    "horizon_days": 30,
                },
                "facts": {
                    "selling_price": -1.0,
                    "purchase_cost": 5.0,
                },
            }
        ],
        "global_facts": {},
        "objective": {
            "sense": "maximize",
            "component_ids": [
                "sales_revenue",
                "purchase_cost",
            ],
        },
        "constraints": [],
    }

    layout = (
        modular_workflow_nodes.build_modular_input_layout(
            specification
        )
    )
    selling_price_input = next(
        item
        for item in layout["product_groups"][0]["fact_inputs"]
        if item["field"]["name"] == "selling_price"
    )

    state = {
        "modular_model": {
            "proposal": {},
            "specification": specification,
            "pending_input_submissions": [
                {
                    "missing_input": selling_price_input,
                    "value": 12.0,
                }
            ],
        }
    }

    result = apply_modular_model_inputs_node(state)

    assert result["modular_model"]["specification"][
        "products"
    ][0]["facts"]["selling_price"] == 12.0
    assert result["modular_model"]["validation"]["is_valid"]
    assert get_workflow_status(result) == (
        WORKFLOW_READY_FOR_REVIEW
    )
