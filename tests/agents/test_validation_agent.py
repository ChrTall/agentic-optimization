from agentic_optimization.agents.validation_agent import (
    detect_model_specific_inputs,
    validate_structured_constraints,
)
from agentic_optimization.state import (
    WORKFLOW_MODEL_INPUTS_INVALID,
    WORKFLOW_NEEDS_MODEL_SPECIFIC_INPUTS,
    WORKFLOW_READY_FOR_REVIEW,
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


def test_unsupported_constraint_is_informational() -> None:
    unsupported_constraint = (
        "Während der Saison kann nicht nachbestellt werden."
    )
    state = {
        "selected_model_family": "newsvendor",
        "selected_model_variant": (
            "single_product_single_period"
        ),
        "model_specific_inputs": {
            "newsvendor": {
                "single_product_single_period": {
                    "selling_price": 80.0,
                    "purchase_cost": 55.0,
                    "salvage_value": 40.0,
                    "demand_scenarios": [
                        5.0,
                        6.0,
                        7.0,
                        8.0,
                    ],
                    "scenario_probabilities": [
                        0.2,
                        0.25,
                        0.3,
                        0.25,
                    ],
                },
            },
        },
        "structured_constraints": [
            {
                "type": "maximum_order_quantity",
                "parameters": {
                    "quantity": 7.0,
                },
                "source_text": (
                    "Höchstens 7 Ornamente bestellen."
                ),
            }
        ],
        "unsupported_constraints": [
            unsupported_constraint,
        ],
    }

    result = validate_structured_constraints(state)

    assert result["constraint_validation_errors"] == []
    assert result["ready_for_optimization"] is True
    assert result["unsupported_constraints"] == [
        unsupported_constraint,
    ]
    assert (
        get_workflow_status(result)
        == WORKFLOW_READY_FOR_REVIEW
    )


def test_derives_required_inputs_from_field_definitions() -> None:
    state = {
        "selected_model_family": "newsvendor",
        "selected_model_variant": (
            "single_product_single_period"
        ),
        "model_specific_inputs": {
            "newsvendor": {
                "single_product_single_period": {},
            },
        },
    }

    result = detect_model_specific_inputs(state)

    assert result["missing_model_input_fields"] == [
        "selling_price",
        "purchase_cost",
        "salvage_value",
        "demand_scenarios",
        "scenario_probabilities",
    ]

    assert (
        get_workflow_status(result)
        == WORKFLOW_NEEDS_MODEL_SPECIFIC_INPUTS
    )
