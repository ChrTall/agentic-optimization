from copy import deepcopy

import pytest

from agentic_optimization.models.specification import (
    ModelSpecification,
)
from agentic_optimization.models.validator import (
    validate_model_specification,
)
from agentic_optimization.models.core_registry import (
    CORE_REGISTRY,
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


def test_disposal_cost_adds_required_product_fact() -> None:
    specification = create_complete_specification()

    specification["objective"]["component_ids"].append(
        "disposal_cost"
    )

    result = validate_model_specification(specification)

    assert result["is_valid"] is False
    assert result["errors"] == []
    assert len(result["missing_inputs"]) == 1

    missing_input = result["missing_inputs"][0]

    assert missing_input["source"] == "product_fact"
    assert missing_input["product_id"] == "product_1"
    assert missing_input["component_id"] == "disposal_cost"
    assert missing_input["field"]["name"] == "disposal_cost"
    assert missing_input["field"]["label"] == (
        "Entsorgungskosten pro unverkaufter Einheit"
    )


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


def test_rejects_non_object_forecast() -> None:
    specification = create_complete_specification()
    specification["products"][0]["forecast"] = "invalid"

    result = validate_model_specification(specification)

    assert result["is_valid"] is False
    assert any(
        "product_1" in error
        and "muss als Objekt vorliegen" in error
        for error in result["errors"]
    )


def test_rejects_non_object_specification() -> None:
    result = validate_model_specification("invalid")

    assert result["is_valid"] is False
    assert result["errors"] == [
        "Die Modellspezifikation muss als Objekt vorliegen."
    ]
    assert result["missing_inputs"] == []


def test_rejects_non_list_products() -> None:
    specification = create_complete_specification()
    specification["products"] = {}

    result = validate_model_specification(specification)

    assert result["is_valid"] is False
    assert any(
        "Produkte müssen als Liste vorliegen" in error
        for error in result["errors"]
    )


def test_rejects_non_object_product() -> None:
    specification = create_complete_specification()
    specification["products"] = ["invalid"]

    result = validate_model_specification(specification)

    assert result["is_valid"] is False
    assert any(
        "Produkt an Position 0" in error
        and "muss als Objekt vorliegen" in error
        for error in result["errors"]
    )


def test_rejects_non_object_product_facts() -> None:
    specification = create_complete_specification()
    specification["products"][0]["facts"] = []

    result = validate_model_specification(specification)

    assert result["is_valid"] is False
    assert any(
        "Produktangaben" in error
        and "müssen als Objekt vorliegen" in error
        for error in result["errors"]
    )


def test_rejects_non_list_objective_components() -> None:
    specification = create_complete_specification()
    specification["objective"]["component_ids"] = (
        "sales_revenue"
    )

    result = validate_model_specification(specification)

    assert result["is_valid"] is False
    assert any(
        "Zielfunktionsbausteine müssen als Liste vorliegen"
        in error
        for error in result["errors"]
    )


def test_rejects_non_object_constraint_parameters() -> None:
    specification = create_complete_specification()
    specification["constraints"][0]["parameters"] = []

    result = validate_model_specification(specification)

    assert result["is_valid"] is False
    assert any(
        "Parameter des Constraints" in error
        and "müssen als Objekt vorliegen" in error
        for error in result["errors"]
    )


def test_rejects_non_string_objective_sense() -> None:
    specification = create_complete_specification()
    specification["objective"]["sense"] = []

    result = validate_model_specification(specification)

    assert result["is_valid"] is False
    assert any(
        "Zielrichtung muss als Zeichenkette vorliegen"
        in error
        for error in result["errors"]
    )


def test_rejects_constraint_number_too_large_for_float() -> None:
    specification = create_complete_specification()
    specification["constraints"][0]["parameters"][
        "quantity"
    ] = 10**1000

    result = validate_model_specification(specification)

    assert result["is_valid"] is False
    assert any(
        "Mindestbestellmenge" in error
        and "muss eine endliche Zahl sein" in error
        for error in result["errors"]
    )


def test_rejects_non_object_objective() -> None:
    specification = create_complete_specification()
    specification["objective"] = []

    result = validate_model_specification(specification)

    assert result["is_valid"] is False
    assert any(
        "Zielfunktion muss als Objekt vorliegen" in error
        for error in result["errors"]
    )


def test_rejects_salvage_value_not_below_purchase_cost() -> None:
    specification = create_complete_specification()

    specification["objective"]["component_ids"].append(
        "salvage_revenue"
    )
    specification["products"][0]["facts"][
        "salvage_value"
    ] = 5.0

    result = validate_model_specification(specification)

    assert result["is_valid"] is False
    assert any(
        "product_1" in error
        and "Restwert" in error
        and "kleiner als die Einkaufskosten" in error
        for error in result["errors"]
    )


def test_accepts_salvage_value_below_purchase_cost() -> None:
    specification = create_complete_specification()

    specification["objective"]["component_ids"].append(
        "salvage_revenue"
    )
    specification["products"][0]["facts"][
        "salvage_value"
    ] = 4.0

    result = validate_model_specification(specification)

    assert result["is_valid"] is True
    assert result["errors"] == []


def test_rejects_objective_without_fulfillment_incentive() -> None:
    specification = create_complete_specification()
    specification["objective"]["component_ids"] = [
        "purchase_cost",
    ]

    result = validate_model_specification(specification)

    assert result["is_valid"] is False
    assert any(
        "keinen positiven Anreiz zur Deckung der Nachfrage"
        in error
        for error in result["errors"]
    )


def test_rejects_salvage_value_that_discourages_sales() -> None:
    specification = create_complete_specification()

    specification["objective"]["component_ids"].append(
        "salvage_revenue"
    )
    specification["products"][0]["facts"][
        "selling_price"
    ] = 4.0
    specification["products"][0]["facts"][
        "salvage_value"
    ] = 4.0

    result = validate_model_specification(specification)

    assert result["is_valid"] is False
    assert any(
        "keinen positiven Anreiz zur Deckung der Nachfrage"
        in error
        for error in result["errors"]
    )


def test_accepts_combined_sales_and_shortage_incentive() -> None:
    specification = create_complete_specification()

    specification["objective"]["component_ids"] = [
        "sales_revenue",
        "salvage_revenue",
        "purchase_cost",
        "shortage_cost",
    ]
    specification["products"][0]["facts"].update(
        {
            "selling_price": 1.0,
            "salvage_value": 4.0,
            "purchase_cost": 5.0,
            "shortage_cost": 4.0,
        }
    )

    result = validate_model_specification(specification)

    assert result["is_valid"] is True
    assert result["errors"] == []


def test_core_required_product_fact_is_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    definition = CORE_REGISTRY[
        "single_period_newsvendor"
    ]["definition"]

    monkeypatch.setitem(
        definition,
        "required_product_facts",
        [
            {
                "name": "lead_time_days",
                "label": "Lieferzeit",
                "type": "integer",
                "required": True,
                "help": "Lieferzeit des Produkts.",
                "min_value": 1,
                "unit": "Tage",
            }
        ],
    )

    result = validate_model_specification(
        create_complete_specification()
    )

    assert result["is_valid"] is False
    assert any(
        missing_input["source"] == "product_fact"
        and missing_input["field"]["name"]
        == "lead_time_days"
        and "component_id" not in missing_input
        for missing_input in result["missing_inputs"]
    )


def test_order_multiple_parameter_is_requested() -> None:
    specification = create_complete_specification()

    specification["constraints"] = [
        {
            "component_id": "order_multiple",
            "product_id": "product_1",
            "parameters": {},
        }
    ]

    result = validate_model_specification(specification)

    assert result["is_valid"] is False
    assert result["errors"] == []
    assert len(result["missing_inputs"]) == 1

    missing_input = result["missing_inputs"][0]

    assert missing_input["source"] == "constraint_parameter"
    assert missing_input["component_id"] == "order_multiple"
    assert missing_input["constraint_index"] == 0
    assert missing_input["field"]["name"] == "multiple"
    assert missing_input["field"]["label"] == "Bestellvielfaches"


def test_rejects_fractional_order_multiple() -> None:
    specification = create_complete_specification()

    specification["constraints"] = [
        {
            "component_id": "order_multiple",
            "product_id": "product_1",
            "parameters": {
                "multiple": 2.5,
            },
        }
    ]

    result = validate_model_specification(specification)

    assert result["is_valid"] is False
    assert any(
        "Bestellvielfaches" in error
        and "muss eine ganze Zahl sein" in error
        for error in result["errors"]
    )


def test_rejects_incompatible_order_multiple_bounds() -> None:
    specification = create_complete_specification()

    specification["constraints"] = [
        {
            "component_id": "minimum_order_quantity",
            "product_id": "product_1",
            "parameters": {
                "quantity": 250.0,
            },
        },
        {
            "component_id": "maximum_order_quantity",
            "product_id": "product_1",
            "parameters": {
                "quantity": 280.0,
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

    result = validate_model_specification(specification)

    assert result["is_valid"] is False
    assert any(
        "product_1" in error
        and "kein zulässiges Bestellvielfaches"
        in error
        for error in result["errors"]
    )
