import pytest

from agentic_optimization.models.execution import (
    execute_model_specification,
)
from agentic_optimization.models.specification import (
    ModelSpecification,
)
from agentic_optimization.models.validator import (
    validate_model_specification,
)


def create_specification() -> ModelSpecification:
    return {
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
        "objective": {
            "sense": "minimize",
            "component_ids": [
                "annual_ordering_cost",
                "annual_holding_cost",
            ],
        },
        "constraints": [],
    }


def test_validates_and_executes_eoq() -> None:
    specification = create_specification()

    validation = validate_model_specification(
        specification
    )
    result = execute_model_specification(specification)

    assert validation["is_valid"] is True
    assert result["status"] == "optimal"
    assert result["solution_method"] == "analytical"
    assert result["objective_value"] == pytest.approx(
        3174.9056603773583
    )
    assert result["decisions"] == [
        {
            "quantity_id": "order_quantity",
            "product_id": "product_1",
            "value": 265.0,
        }
    ]

    technical_representation = result[
        "technical_representation"
    ]

    assert (
        "Mindestbestellmenge = nicht gesetzt"
        in technical_representation
    )
    assert (
        "Höchstbestellmenge = nicht gesetzt"
        in technical_representation
    )
    assert "None" not in technical_representation


def test_reports_missing_holding_cost() -> None:
    specification = create_specification()
    del specification["products"][0]["facts"][
        "holding_cost"
    ]

    validation = validate_model_specification(
        specification
    )

    assert validation["is_valid"] is False
    assert validation["errors"] == []
    assert any(
        missing_input["field"]["name"] == "holding_cost"
        for missing_input in validation["missing_inputs"]
    )


def test_applies_minimum_order_quantity() -> None:
    specification = create_specification()
    specification["constraints"] = [
        {
            "component_id": "minimum_order_quantity",
            "product_id": "product_1",
            "parameters": {
                "quantity": 300.0,
            },
        }
    ]

    result = execute_model_specification(specification)

    assert result["decisions"][0]["value"] == pytest.approx(
        300.0
    )
    assert result["objective_value"] == pytest.approx(
        3200.0
    )


def test_rejects_missing_required_objective_component() -> None:
    specification = create_specification()
    specification["objective"]["component_ids"] = [
        "annual_ordering_cost",
    ]

    validation = validate_model_specification(
        specification
    )

    assert validation["is_valid"] is False
    assert any(
        "annual_holding_cost" in error
        and "benötigt" in error
        for error in validation["errors"]
    )


def test_applies_order_multiple() -> None:
    specification = create_specification()

    specification["constraints"] = [
        {
            "component_id": "order_multiple",
            "product_id": "product_1",
            "parameters": {
                "multiple": 100,
            },
        }
    ]

    validation = validate_model_specification(
        specification
    )
    result = execute_model_specification(specification)

    assert validation["is_valid"] is True
    assert result["status"] == "optimal"
    assert result["solution_method"] == "analytical"
    assert result["decisions"][0]["value"] == 300.0
    assert result["objective_value"] == pytest.approx(
        3200.0
    )


def test_rejects_order_multiple_above_maximum() -> None:
    specification = create_specification()

    specification["constraints"] = [
        {
            "component_id": "maximum_order_quantity",
            "product_id": "product_1",
            "parameters": {
                "quantity": 50.0,
            },
        },
        {
            "component_id": "order_multiple",
            "product_id": "product_1",
            "parameters": {
                "multiple": 100,
            },
        },
    ]

    validation = validate_model_specification(
        specification
    )

    assert validation["is_valid"] is False
    assert any(
        "kein positives zulässiges Bestellvielfaches"
        in error
        for error in validation["errors"]
    )
