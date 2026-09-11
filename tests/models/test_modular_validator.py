from copy import deepcopy

import pytest

from agentic_optimization.models.specification import (
    ModelSpecification,
)
from agentic_optimization.models.validator import (
    validate_model_specification,
)


def create_complete_specification() -> ModelSpecification:
    """Erzeugt eine vollständige Newsvendor-Spezifikation."""

    return {
        "core_id": "single_period_newsvendor",
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
                    "selling_price": 12.0,
                    "purchase_cost": 5.0,
                },
            }
        ],
        "global_facts": {},
        "objective": {
            "sense": "maximize",
            "component_ids": [
                "sales_revenue",
                "purchase_cost",
            ],
        },
        "constraints": [
            {
                "component_id": "minimum_order_quantity",
                "product_id": "product_1",
                "parameters": {
                    "quantity": 30.0,
                },
            }
        ],
    }


def test_complete_specification_is_valid() -> None:
    specification = create_complete_specification()

    result = validate_model_specification(specification)

    assert result["is_valid"] is True
    assert result["errors"] == []
    assert result["missing_inputs"] == []


def test_missing_product_fact_is_reported() -> None:
    specification = create_complete_specification()
    del specification["products"][0]["facts"]["selling_price"]

    result = validate_model_specification(specification)

    assert result["is_valid"] is False
    assert result["errors"] == []

    assert len(result["missing_inputs"]) == 1

    missing_input = result["missing_inputs"][0]

    assert missing_input["source"] == "product_fact"
    assert missing_input["product_id"] == "product_1"
    assert missing_input["component_id"] == "sales_revenue"
    assert missing_input["field"]["name"] == "selling_price"
    assert (
        missing_input["field"]["label"]
        == "Verkaufspreis pro Einheit"
    )


def test_missing_constraint_parameter_is_reported() -> None:
    specification = create_complete_specification()
    specification["constraints"][0]["parameters"] = {}

    result = validate_model_specification(specification)

    assert result["is_valid"] is False
    assert result["errors"] == []

    assert len(result["missing_inputs"]) == 1

    missing_input = result["missing_inputs"][0]

    assert missing_input["source"] == "constraint_parameter"
    assert missing_input["component_id"] == "minimum_order_quantity"
    assert missing_input["constraint_index"] == 0
    assert missing_input["field"]["name"] == "quantity"
    assert missing_input["field"]["label"] == "Mindestbestellmenge"


def test_missing_forecast_value_is_reported() -> None:
    specification = deepcopy(
        create_complete_specification()
    )
    del specification["products"][0]["forecast"]["expected_value"]

    result = validate_model_specification(specification)

    assert result["is_valid"] is False
    assert result["errors"] == []

    assert len(result["missing_inputs"]) == 1

    missing_input = result["missing_inputs"][0]

    assert missing_input["source"] == "forecast"
    assert missing_input["product_id"] == "product_1"
    assert missing_input["field"]["name"] == "expected_value"
    assert missing_input["field"]["label"] == "Erwartete Nachfrage"


def test_selected_component_adds_required_input() -> None:
    specification = create_complete_specification()

    specification["objective"]["component_ids"].append(
        "salvage_revenue"
    )

    result = validate_model_specification(specification)

    assert result["is_valid"] is False
    assert result["errors"] == []

    assert len(result["missing_inputs"]) == 1

    missing_input = result["missing_inputs"][0]

    assert missing_input["source"] == "product_fact"
    assert missing_input["component_id"] == "salvage_revenue"
    assert missing_input["field"]["name"] == "salvage_value"


def test_missing_fact_required_by_multiple_components_is_unique() -> None:
    specification = create_complete_specification()
    del specification["products"][0]["facts"]["purchase_cost"]

    specification["constraints"] = [
        {
            "component_id": "budget_limit",
            "parameters": {
                "budget": 500.0,
            },
        }
    ]

    result = validate_model_specification(specification)

    purchase_cost_inputs = [
        missing_input
        for missing_input in result["missing_inputs"]
        if (
            missing_input["field"]["name"]
            == "purchase_cost"
        )
    ]

    assert len(purchase_cost_inputs) == 1


def test_shortage_cost_requires_product_fact() -> None:
    specification = create_complete_specification()

    specification["objective"]["component_ids"].append(
        "shortage_cost"
    )

    result = validate_model_specification(specification)

    assert result["is_valid"] is False
    assert result["errors"] == []

    assert len(result["missing_inputs"]) == 1

    missing_input = result["missing_inputs"][0]

    assert missing_input["component_id"] == "shortage_cost"
    assert missing_input["field"]["name"] == "shortage_cost"
    assert missing_input["product_id"] == "product_1"


def test_storage_capacity_requires_space_for_each_product() -> None:
    specification = create_complete_specification()

    second_product = deepcopy(
        specification["products"][0]
    )
    second_product["product_id"] = "product_2"

    specification["products"].append(second_product)

    specification["constraints"] = [
        {
            "component_id": "storage_capacity",
            "parameters": {
                "capacity": 210.0,
            },
        }
    ]

    result = validate_model_specification(specification)

    storage_inputs = [
        missing_input
        for missing_input in result["missing_inputs"]
        if (
            missing_input["field"]["name"]
            == "storage_space_per_unit"
        )
    ]

    assert len(storage_inputs) == 2

    assert {
        missing_input["product_id"]
        for missing_input in storage_inputs
    } == {
        "product_1",
        "product_2",
    }


def test_rejects_duplicate_objective_component() -> None:
    specification = create_complete_specification()

    specification["objective"]["component_ids"].append(
        "sales_revenue"
    )

    result = validate_model_specification(specification)

    assert result["is_valid"] is False
    assert any(
        "darf nur einmal ausgewählt werden" in error
        for error in result["errors"]
    )


def test_rejects_duplicate_constraint_for_same_product() -> None:
    specification = create_complete_specification()

    specification["constraints"] = [
        {
            "component_id": "minimum_order_quantity",
            "product_id": "product_1",
            "parameters": {
                "quantity": 20.0,
            },
        },
        {
            "component_id": "minimum_order_quantity",
            "product_id": "product_1",
            "parameters": {
                "quantity": 30.0,
            },
        },
    ]

    result = validate_model_specification(specification)

    assert result["is_valid"] is False
    assert any(
        "mehrfach angegeben" in error
        for error in result["errors"]
    )


def test_rejects_minimum_above_maximum() -> None:
    specification = create_complete_specification()

    specification["constraints"] = [
        {
            "component_id": "minimum_order_quantity",
            "product_id": "product_1",
            "parameters": {
                "quantity": 100.0,
            },
        },
        {
            "component_id": "maximum_order_quantity",
            "product_id": "product_1",
            "parameters": {
                "quantity": 80.0,
            },
        },
    ]

    result = validate_model_specification(specification)

    assert result["is_valid"] is False
    assert any(
        "Mindestbestellmenge die Höchstbestellmenge" in error
        for error in result["errors"]
    )


def test_unregistered_core_is_rejected() -> None:
    specification = create_complete_specification()
    specification["core_id"] = "unknown_core"

    result = validate_model_specification(specification)

    assert result["is_valid"] is False
    assert result["errors"] == [
        "Das Grundmodell 'unknown_core' wird nicht unterstützt."
    ]
    assert result["missing_inputs"] == []


def test_rejects_non_numeric_product_fact() -> None:
    specification = create_complete_specification()
    specification["products"][0]["facts"]["selling_price"] = "zwölf"

    result = validate_model_specification(specification)

    assert result["is_valid"] is False
    assert any(
        "Verkaufspreis pro Einheit" in error
        and "muss eine Zahl sein" in error
        for error in result["errors"]
    )


def test_rejects_non_integer_forecast_horizon() -> None:
    specification = create_complete_specification()
    specification["products"][0]["forecast"]["horizon_days"] = 30.5

    result = validate_model_specification(specification)

    assert result["is_valid"] is False
    assert any(
        "Prognosezeitraum" in error
        and "muss eine ganze Zahl sein" in error
        for error in result["errors"]
    )


def test_rejects_constraint_value_below_minimum() -> None:
    specification = create_complete_specification()
    specification["constraints"][0]["parameters"]["quantity"] = -1.0

    result = validate_model_specification(specification)

    assert result["is_valid"] is False
    assert any(
        "Mindestbestellmenge" in error
        and "muss mindestens 0.0 betragen" in error
        for error in result["errors"]
    )


def test_rejects_non_monotonic_forecast_quantiles() -> None:
    specification = create_complete_specification()
    specification["products"][0]["forecast"]["quantiles"] = [
        {
            "level": 0.1,
            "value": 100.0,
        },
        {
            "level": 0.9,
            "value": 80.0,
        },
    ]

    result = validate_model_specification(specification)

    assert result["is_valid"] is False
    assert any(
        "monoton steigen" in error
        for error in result["errors"]
    )


def test_validates_constraint_fact_for_every_product() -> None:
    specification = create_complete_specification()

    second_product = deepcopy(
        specification["products"][0]
    )
    second_product["product_id"] = "product_2"

    specification["products"].append(second_product)

    specification["products"][0]["facts"][
        "storage_space_per_unit"
    ] = -1.0

    specification["products"][1]["facts"][
        "storage_space_per_unit"
    ] = 2.0

    specification["constraints"] = [
        {
            "component_id": "storage_capacity",
            "parameters": {
                "capacity": 500.0,
            },
        }
    ]

    result = validate_model_specification(specification)

    assert result["is_valid"] is False
    assert any(
        "Platzbedarf pro Einheit" in error
        and "muss mindestens 0.0 betragen" in error
        and "product_1" in error
        for error in result["errors"]
    )


@pytest.mark.parametrize(
    "invalid_value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_rejects_non_finite_product_fact(
    invalid_value: float,
) -> None:
    specification = create_complete_specification()
    specification["products"][0]["facts"][
        "selling_price"
    ] = invalid_value

    result = validate_model_specification(specification)

    assert result["is_valid"] is False
    assert any(
        "Verkaufspreis pro Einheit" in error
        and "muss eine endliche Zahl sein" in error
        for error in result["errors"]
    )