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

    assert prepared_result["selected_model"] == {
        "family_id": "newsvendor",
        "family_name": "Newsvendor",
        "variant_id": "single_product_single_period",
        "variant_name": "Single Product, Single Period",
    }

    assert "model_config" not in prepared_result


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