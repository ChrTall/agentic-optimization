from unittest.mock import Mock

import pytest

from agentic_optimization.graph import build_graph
from agentic_optimization.state import (
    EVENT_MODEL_INPUTS_SUBMITTED,
    EVENT_OPTIMIZATION_REQUESTED,
    WORKFLOW_READY_FOR_RESULT,
    WORKFLOW_READY_FOR_REVIEW,
    get_workflow_status,
)


@pytest.fixture
def valid_inputs() -> dict:
    return {
        "selling_price": 10.0,
        "purchase_cost": 5.0,
        "salvage_value": 2.0,
        "demand_scenarios": [
            80.0,
            100.0,
            120.0,
        ],
        "scenario_probabilities": [
            0.25,
            0.5,
            0.25,
        ],
    }


@pytest.fixture
def graph():
    # Für diese Tests wird das LLM nicht aufgerufen, weil der
    # Workflow direkt bei der Modellvalidierung beziehungsweise
    # Optimierung startet.
    unused_llm = Mock()

    return build_graph(unused_llm)


def create_model_submission_state(
    valid_inputs: dict,
) -> dict:
    return {
        "control": {
            "event": EVENT_MODEL_INPUTS_SUBMITTED,
        },
        "selected_model_family": "newsvendor",
        "selected_model_variant": (
            "single_product_single_period"
        ),
        "model_config": {
            "family_id": "newsvendor",
            "family_name": "Newsvendor",
            "variant_id": (
                "single_product_single_period"
            ),
            "variant_name": (
                "Single Product, Single Period"
            ),
            "decision": "Bestellmenge",
            "objective": "Gewinn maximieren",
        },
        "requirements": {
            "decision": "Bestellmenge",
            "objective": "Gewinn maximieren",
            "demand_type": "Unsicher",
            "constraints": [],
            "raw_description": "",
        },
        "model_specific_inputs": {
            "newsvendor": {
                "single_product_single_period": (
                    valid_inputs
                ),
            }
        },
        "structured_constraints": [],
        "unsupported_constraints": [],
        "constraint_mapping_errors": [],
        "review_confirmed": False,
    }


def create_q_r_submission_state() -> dict:
    return {
        "control": {
            "event": EVENT_MODEL_INPUTS_SUBMITTED,
        },
        "selected_model_family": "reorder_point",
        "selected_model_variant": "q_r_policy",
        "model_config": {
            "family_id": "reorder_point",
            "family_name": "Reorder Point Policies",
            "variant_id": "q_r_policy",
            "variant_name": "Q-R Policy",
            "decision": "order_quantity",
            "objective": "minimize_cost",
        },
        "requirements": {
            "decision": "order_quantity",
            "objective": "minimize_cost",
            "demand_type": "uncertain",
            "product_scope": "single_product",
            "planning_horizon": "multiple_periods",
            "review_type": "continuous_review",
            "constraints": [],
            "raw_description": "",
        },
        "model_specific_inputs": {
            "reorder_point": {
                "q_r_policy": {
                    "annual_demand": 10_000.0,
                    "fixed_order_cost": 100.0,
                    "holding_cost": 5.0,
                    "penalty_cost": 25.0,
                    "lead_time_demand_mean": 400.0,
                    "lead_time_demand_std": 50.0,
                },
            },
        },
        "structured_constraints": [
            {
                "type": "storage_capacity",
                "parameters": {
                    "capacity": 2_000.0,
                },
                "source_text": (
                    "Die Bestandsposition darf 2000 "
                    "Einheiten nicht überschreiten."
                ),
            }
        ],
        "unsupported_constraints": [],
        "constraint_mapping_errors": [],
        "review_confirmed": False,
    }


def create_eoq_model_submission_state(
    constraints: list[dict] | None = None,
) -> dict:
    return {
        "control": {
            "event": EVENT_MODEL_INPUTS_SUBMITTED,
        },
        "selected_model_family": "eoq",
        "selected_model_variant": "basic_eoq",
        "requirements": {
            "decision": "order_quantity",
            "objective": "minimize_cost",
            "demand_type": "known",
            "product_scope": "single_product",
            "planning_horizon": "repeated_cycles",
            "constraints": [],
            "raw_description": "",
        },
        "model_specific_inputs": {
            "eoq": {
                "basic_eoq": {
                    "annual_demand": 3500.0,
                    "ordering_cost": 120.0,
                    "holding_cost": 12.0,
                },
            }
        },
        "structured_constraints": (
            constraints or []
        ),
        "unsupported_constraints": [],
        "constraint_mapping_errors": [],
        "review_confirmed": False,
    }


def test_model_submission_stops_at_review(
    graph,
    valid_inputs: dict,
) -> None:
    state = create_model_submission_state(
        valid_inputs
    )

    result = graph.invoke(state)

    assert (
        get_workflow_status(result)
        == WORKFLOW_READY_FOR_REVIEW
    )

    assert result["ready_for_optimization"] is True
    assert result["review_confirmed"] is False

    assert result.get("optimization_result") is None
    assert result.get("result") is None


def test_unsupported_constraint_does_not_block_review_or_solver(
    graph,
    valid_inputs: dict,
) -> None:
    state = create_model_submission_state(
        valid_inputs
    )
    state["structured_constraints"] = [
        {
            "type": "maximum_order_quantity",
            "parameters": {
                "quantity": 110.0,
            },
            "source_text": (
                "Höchstens 110 Einheiten bestellen."
            ),
        }
    ]
    state["unsupported_constraints"] = [
        "Während der Saison kann nicht nachbestellt werden.",
    ]

    review_state = graph.invoke(state)

    assert (
        get_workflow_status(review_state)
        == WORKFLOW_READY_FOR_REVIEW
    )
    assert review_state["constraint_validation_errors"] == []
    assert review_state["ready_for_optimization"] is True

    result = graph.invoke(
        {
            **review_state,
            "control": {
                **review_state.get("control", {}),
                "event": EVENT_OPTIMIZATION_REQUESTED,
            },
            "review_confirmed": True,
        }
    )

    assert (
        get_workflow_status(result)
        == WORKFLOW_READY_FOR_RESULT
    )
    assert result["optimization_errors"] == []
    assert result["unsupported_constraints"] == [
        "Während der Saison kann nicht nachbestellt werden.",
    ]
    assert result["optimization_result"][
        "order_quantity"
    ] == pytest.approx(100.0)


def test_q_r_submission_reaches_review_and_result(
    graph,
) -> None:
    review_state = graph.invoke(
        create_q_r_submission_state()
    )

    assert (
        get_workflow_status(review_state)
        == WORKFLOW_READY_FOR_REVIEW
    )
    assert review_state["ready_for_optimization"] is True

    optimization_state = {
        **review_state,
        "control": {
            **review_state.get("control", {}),
            "event": EVENT_OPTIMIZATION_REQUESTED,
        },
        "review_confirmed": True,
    }

    result = graph.invoke(optimization_state)

    assert (
        get_workflow_status(result)
        == WORKFLOW_READY_FOR_RESULT
    )
    assert result["optimization_errors"] == []
    assert result["optimization_result"]["solver_status"] == "Optimal"
    assert isinstance(result["optimization_result"]["order_quantity"], int)
    assert isinstance(result["optimization_result"]["reorder_point"], int)
    assert isinstance(result["optimization_result"]["safety_stock"], int)


def test_confirmed_review_starts_optimization(
    graph,
    valid_inputs: dict,
) -> None:
    state = create_model_submission_state(
        valid_inputs
    )

    review_state = graph.invoke(state)

    optimization_state = {
        **review_state,
        "control": {
            **review_state.get("control", {}),
            "event": EVENT_OPTIMIZATION_REQUESTED,
        },
        "review_confirmed": True,
    }

    result = graph.invoke(
        optimization_state
    )

    assert (
        get_workflow_status(result)
        == WORKFLOW_READY_FOR_RESULT
    )

    optimization_result = result[
        "optimization_result"
    ]

    assert optimization_result is not None

    assert (
        optimization_result["solver_status"]
        == "Optimal"
    )

    assert optimization_result[
        "order_quantity"
    ] == pytest.approx(100.0)

    assert result["optimization_errors"] == []

    prepared_result = result["result"]

    assert (
        prepared_result["optimization_result"]
        ["solver_status"]
        == "Optimal"
    )


def test_review_with_constraint_affects_result(
    graph,
    valid_inputs: dict,
) -> None:
    state = create_model_submission_state(
        valid_inputs
    )

    state["structured_constraints"] = [
        {
            "type": "maximum_order_quantity",
            "parameters": {
                "quantity": 90.0,
            },
            "source_text": (
                "Es dürfen höchstens 90 Einheiten "
                "bestellt werden."
            ),
        }
    ]

    review_state = graph.invoke(state)

    assert (
        get_workflow_status(review_state)
        == WORKFLOW_READY_FOR_REVIEW
    )

    optimization_state = {
        **review_state,
        "control": {
            **review_state.get("control", {}),
            "event": EVENT_OPTIMIZATION_REQUESTED,
        },
        "review_confirmed": True,
    }

    result = graph.invoke(
        optimization_state
    )

    optimization_result = result[
        "optimization_result"
    ]

    assert optimization_result is not None

    assert optimization_result[
        "order_quantity"
    ] == pytest.approx(90.0)


def test_optimization_without_confirmation_returns_error_result(
    graph,
    valid_inputs: dict,
) -> None:
    state = create_model_submission_state(
        valid_inputs
    )

    review_state = graph.invoke(state)

    optimization_state = {
        **review_state,
        "control": {
            **review_state.get("control", {}),
            "event": EVENT_OPTIMIZATION_REQUESTED,
        },
        "review_confirmed": False,
    }

    result = graph.invoke(
        optimization_state
    )

    assert (
        get_workflow_status(result)
        == WORKFLOW_READY_FOR_RESULT
    )

    assert result["optimization_result"] is None
    assert result["optimization_errors"]

    assert any(
        "nicht durch den Nutzer bestätigt"
        in error
        for error in result["optimization_errors"]
    )


def test_basic_eoq_review_and_optimization_flow(
    graph,
) -> None:
    state = create_eoq_model_submission_state()

    review_state = graph.invoke(state)

    assert (
        get_workflow_status(review_state)
        == WORKFLOW_READY_FOR_REVIEW
    )

    assert review_state[
        "ready_for_optimization"
    ] is True

    optimization_state = {
        **review_state,
        "control": {
            **review_state.get("control", {}),
            "event": EVENT_OPTIMIZATION_REQUESTED,
        },
        "review_confirmed": True,
    }

    result = graph.invoke(
        optimization_state
    )

    assert (
        get_workflow_status(result)
        == WORKFLOW_READY_FOR_RESULT
    )

    optimization_result = result[
        "optimization_result"
    ]

    assert optimization_result is not None

    assert (
        optimization_result["solver_status"]
        == "Optimal"
    )

    assert optimization_result[
        "order_quantity"
    ] == pytest.approx(
        math.sqrt(70000)
    )

    assert result["optimization_errors"] == []

    assert result["result"][
        "selected_model"
    ] == {
        "family_id": "eoq",
        "family_name": "EOQ",
        "variant_id": "basic_eoq",
        "variant_name": "Basic EOQ",
    }


def test_basic_eoq_minimum_quantity_affects_result(
    graph,
) -> None:
    constraints = [
        {
            "type": "minimum_order_quantity",
            "parameters": {
                "quantity": 300.0,
            },
            "source_text": None,
        }
    ]

    state = create_eoq_model_submission_state(
        constraints
    )

    review_state = graph.invoke(state)

    assert (
        get_workflow_status(review_state)
        == WORKFLOW_READY_FOR_REVIEW
    )

    optimization_state = {
        **review_state,
        "control": {
            **review_state.get("control", {}),
            "event": EVENT_OPTIMIZATION_REQUESTED,
        },
        "review_confirmed": True,
    }

    result = graph.invoke(
        optimization_state
    )

    assert (
        get_workflow_status(result)
        == WORKFLOW_READY_FOR_RESULT
    )

    optimization_result = result[
        "optimization_result"
    ]

    assert optimization_result is not None
    assert (
        optimization_result["order_quantity"]
        == 300.0
    )
    assert optimization_result[
        "annual_ordering_cost"
    ] == pytest.approx(1400.0)
    assert optimization_result[
        "annual_holding_cost"
    ] == pytest.approx(1800.0)
    assert optimization_result[
        "annual_relevant_cost"
    ] == pytest.approx(3200.0)
