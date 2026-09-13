import json
from unittest.mock import Mock

from langchain_core.messages import AIMessage

import pytest
from agentic_optimization.agents.modular_workflow_nodes import (
    apply_modular_model_inputs_node,
    create_modular_model_preparation_node,
    execute_modular_model_node,
)
from agentic_optimization.state import (
    WORKFLOW_NEEDS_MODEL_SPECIFIC_INPUTS,
    WORKFLOW_READY_FOR_REVIEW,
    WORKFLOW_OPTIMIZATION_COMPLETED,
    get_workflow_status,
)


def test_prepares_modular_model_in_graph_state() -> None:
    response = {
        "core_id": "continuous_review_q_r",
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