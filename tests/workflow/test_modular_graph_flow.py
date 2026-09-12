import json
from unittest.mock import Mock

from langchain_core.messages import AIMessage

from agentic_optimization.graph import build_graph
from agentic_optimization.state import (
    EVENT_MODULAR_INPUTS_SUBMITTED,
    EVENT_MODULAR_MODEL_PREPARATION_REQUESTED,
    EVENT_MODULAR_OPTIMIZATION_REQUESTED,
    WORKFLOW_NEEDS_MODEL_SPECIFIC_INPUTS,
    WORKFLOW_OPTIMIZATION_COMPLETED,
    WORKFLOW_READY_FOR_REVIEW,
    get_workflow_status,
)


def test_runs_modular_model_through_graph() -> None:
    response = {
        "core_id": "continuous_review_q_r",
        "objective_sense": "minimize",
        "objective_components": [
            {
                "component_id": "annual_ordering_cost",
                "reason": "Bestellkosten berücksichtigen.",
                "sources": ["source_1"],
            },
            {
                "component_id": "annual_holding_cost",
                "reason": "Lagerkosten berücksichtigen.",
                "sources": ["source_1"],
            },
            {
                "component_id": "annual_shortage_cost",
                "reason": "Fehlmengenkosten berücksichtigen.",
                "sources": ["source_1"],
            },
        ],
        "constraints": [],
    }

    llm = Mock()
    llm.invoke.return_value = AIMessage(
        content=json.dumps(response)
    )

    graph = build_graph(llm)

    initial_state = {
        "control": {
            "event": (
                EVENT_MODULAR_MODEL_PREPARATION_REQUESTED
            ),
        },
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

    prepared_state = graph.invoke(initial_state)
    prepared_model = prepared_state["modular_model"]

    assert get_workflow_status(prepared_state) == (
        WORKFLOW_NEEDS_MODEL_SPECIFIC_INPUTS
    )
    assert "event" not in prepared_state["control"]

    values = {
        "ordering_cost": 50.0,
        "holding_cost": 2.0,
        "shortage_cost": 20.0,
    }

    input_state = {
        **prepared_state,
        "control": {
            **prepared_state["control"],
            "event": EVENT_MODULAR_INPUTS_SUBMITTED,
        },
        "modular_model": {
            **prepared_model,
            "pending_input_submissions": [
                {
                    "missing_input": missing_input,
                    "value": values[
                        missing_input["field"]["name"]
                    ],
                }
                for missing_input
                in prepared_model["validation"][
                    "missing_inputs"
                ]
            ],
        },
    }

    completed_state = graph.invoke(input_state)
    completed_model = completed_state["modular_model"]

    assert completed_model["validation"]["is_valid"] is True
    assert get_workflow_status(completed_state) == (
        WORKFLOW_READY_FOR_REVIEW
    )
    assert "event" not in completed_state["control"]

    optimization_state = {
        **completed_state,
        "control": {
            **completed_state["control"],
            "event": EVENT_MODULAR_OPTIMIZATION_REQUESTED,
        },
        "modular_model": {
            **completed_model,
            "review_confirmed": True,
        },
    }

    optimized_state = graph.invoke(optimization_state)
    optimization_result = optimized_state[
        "modular_model"
    ]["optimization_result"]

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
    assert "event" not in optimized_state["control"]

    llm.invoke.assert_called_once()
