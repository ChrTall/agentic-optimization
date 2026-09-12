from agentic_optimization.state import (
    EVENT_MODEL_INPUTS_SUBMITTED,
    invalidate_optimization_state,
)


def test_invalidates_review_and_optimization_results() -> None:
    state = {
        "control": {
            "event": EVENT_MODEL_INPUTS_SUBMITTED,
            "workflow_status": "ready_for_result",
        },
        "review_confirmed": True,
        "ready_for_optimization": True,
        "optimization_result": {
            "solver_status": "Optimal",
            "order_quantity": 100,
            "model_representation": "old model",
        },
        "optimization_errors": [
            "old error",
        ],
        "result": {
            "summary": "old result",
        },
        "structured_constraints": [
            {
                "type": "budget_limit",
                "parameters": {
                    "budget": 5000,
                },
                "source_text": "Maximales Budget 5000 Euro",
            }
        ],
    }

    result = invalidate_optimization_state(state)

    assert result["review_confirmed"] is False
    assert result["ready_for_optimization"] is False
    assert result["optimization_result"] is None
    assert result["optimization_errors"] == []
    assert result["result"] == {}

    # Fachliche Eingaben dürfen nicht gelöscht werden.
    assert result["structured_constraints"] == (
        state["structured_constraints"]
    )

    # Das auslösende Event bleibt erhalten.
    assert (
        result["control"]["event"]
        == EVENT_MODEL_INPUTS_SUBMITTED
    )


def test_invalidation_does_not_mutate_original_state() -> None:
    original_state = {
        "review_confirmed": True,
        "ready_for_optimization": True,
        "optimization_result": {
            "solver_status": "Optimal",
        },
        "optimization_errors": [],
        "result": {
            "summary": "old result",
        },
    }

    invalidated_state = invalidate_optimization_state(
        original_state
    )

    assert invalidated_state is not original_state

    assert original_state["review_confirmed"] is True
    assert original_state["optimization_result"] == {
        "solver_status": "Optimal",
    }

    assert invalidated_state["review_confirmed"] is False
    assert invalidated_state["optimization_result"] is None


def test_invalidates_modular_optimization_result() -> None:
    state = {
        "modular_model": {
            "core_id": "eoq",
            "proposal": {
                "core_id": "eoq",
                "objective_sense": "minimize",
                "objective_components": [],
                "constraints": [],
            },
            "review_confirmed": True,
            "optimization_result": {
                "status": "optimal",
                "solution_method": "analytical",
                "objective_value": 100.0,
                "decisions": [],
                "metrics": [],
                "technical_representation": None,
                "errors": [],
            },
        },
    }

    result = invalidate_optimization_state(state)

    assert result["modular_model"][
        "review_confirmed"
    ] is False
    assert result["modular_model"][
        "optimization_result"
    ] is None

    assert result["modular_model"]["core_id"] == "eoq"
    assert "proposal" in result["modular_model"]

    assert state["modular_model"][
        "review_confirmed"
    ] is True
    assert state["modular_model"][
        "optimization_result"
    ] is not None
