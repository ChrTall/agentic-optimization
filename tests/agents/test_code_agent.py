import pytest

from agentic_optimization.agents.code_agent import (
    optimize_model,
)
from agentic_optimization.state import (
    WORKFLOW_OPTIMIZATION_COMPLETED,
    WORKFLOW_OPTIMIZATION_FAILED,
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


def test_rejects_optimization_before_validation() -> None:
    state = {
        "ready_for_optimization": False,
        "review_confirmed": True,
    }

    result = optimize_model(state)

    assert (
        get_workflow_status(result)
        == WORKFLOW_OPTIMIZATION_FAILED
    )
    assert result["optimization_result"] is None
    assert result["optimization_errors"]


def test_rejects_optimization_without_review_confirmation() -> None:
    state = {
        "ready_for_optimization": True,
        "review_confirmed": False,
    }

    result = optimize_model(state)

    assert (
        get_workflow_status(result)
        == WORKFLOW_OPTIMIZATION_FAILED
    )
    assert result["optimization_result"] is None

    assert any(
        "nicht durch den Nutzer bestätigt" in error
        for error in result["optimization_errors"]
    )


def test_rejects_missing_model_selection() -> None:
    state = {
        "ready_for_optimization": True,
        "review_confirmed": True,
    }

    result = optimize_model(state)

    assert (
        get_workflow_status(result)
        == WORKFLOW_OPTIMIZATION_FAILED
    )
    assert result["optimization_result"] is None
    assert result["optimization_errors"]


def test_rejects_unknown_model_variant() -> None:
    state = {
        "ready_for_optimization": True,
        "review_confirmed": True,
        "selected_model_family": "newsvendor",
        "selected_model_variant": "unknown_variant",
    }

    result = optimize_model(state)

    assert (
        get_workflow_status(result)
        == WORKFLOW_OPTIMIZATION_FAILED
    )
    assert result["optimization_result"] is None

    assert any(
        "ist nicht registriert" in error
        for error in result["optimization_errors"]
    )


def test_optimizes_registered_variant(
    valid_inputs: dict,
) -> None:
    state = {
        "ready_for_optimization": True,
        "review_confirmed": True,
        "selected_model_family": "newsvendor",
        "selected_model_variant": (
            "single_product_single_period"
        ),
        "model_specific_inputs": {
            "newsvendor": {
                "single_product_single_period": (
                    valid_inputs
                ),
            }
        },
        "structured_constraints": [],
    }

    result = optimize_model(state)

    assert (
        get_workflow_status(result)
        == WORKFLOW_OPTIMIZATION_COMPLETED
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


def test_passes_constraints_to_registered_optimizer(
    valid_inputs: dict,
) -> None:
    state = {
        "ready_for_optimization": True,
        "review_confirmed": True,
        "selected_model_family": "newsvendor",
        "selected_model_variant": (
            "single_product_single_period"
        ),
        "model_specific_inputs": {
            "newsvendor": {
                "single_product_single_period": (
                    valid_inputs
                ),
            }
        },
        "structured_constraints": [
            {
                "type": "maximum_order_quantity",
                "parameters": {
                    "quantity": 90.0,
                },
                "source_text": None,
            }
        ],
    }

    result = optimize_model(state)

    assert (
        get_workflow_status(result)
        == WORKFLOW_OPTIMIZATION_COMPLETED
    )

    optimization_result = result[
        "optimization_result"
    ]

    assert optimization_result is not None

    assert optimization_result[
        "order_quantity"
    ] == pytest.approx(90.0)