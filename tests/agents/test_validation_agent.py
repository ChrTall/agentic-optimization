from agentic_optimization.agents.validation_agent import (
    validate_structured_constraints,
)
from agentic_optimization.state import (
    WORKFLOW_MODEL_INPUTS_INVALID,
    get_workflow_status,
)


def test_rejects_active_constraint_without_required_parameter() -> None:
    state = {
        "selected_model_family": "newsvendor",
        "selected_model_variant": (
            "single_product_single_period"
        ),
        "model_specific_inputs": {
            "newsvendor": {
                "single_product_single_period": {
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
                },
            },
        },
        "structured_constraints": [
            {
                "type": "minimum_order_quantity",
                "parameters": {},
                "source_text": (
                    "Mindestbestellmenge = 40 Stück"
                ),
            }
        ],
        "unsupported_constraints": [],
    }

    result = validate_structured_constraints(
        state
    )

    assert result["constraint_validation_errors"]

    assert any(
        "fehlt der Parameter 'quantity'"
        in error
        for error in result[
            "constraint_validation_errors"
        ]
    )

    assert result["ready_for_optimization"] is False

    assert (
        get_workflow_status(result)
        == WORKFLOW_MODEL_INPUTS_INVALID
    )