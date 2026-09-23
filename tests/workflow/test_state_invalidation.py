from agentic_optimization.state import invalidate_optimization_state


def test_invalidates_modular_optimization_result() -> None:
    state = {
        "modular_model": {
            "core_id": "eoq",
            "proposal": {
                "core_id": "eoq",
                "selection_status": "complete",
                "objective_sense": "minimize",
                "objective_components": [],
                "constraints": [],
                "unresolved_requirements": [],
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
