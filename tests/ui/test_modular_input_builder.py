from agentic_optimization.models.specification import (
    ModelSpecification,
    ProductSpecification,
)
from agentic_optimization.models.validator import (
    validate_model_specification,
)
from agentic_optimization.ui.modular_input_builder import (
    apply_modular_input_value,
    build_modular_input_layout,
)

def create_product(
    product_id: str,
) -> ProductSpecification:
    return {
        "product_id": product_id,
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


def create_specification(
    products: list[ProductSpecification],
) -> ModelSpecification:
    return {
        "core_id": "single_period_newsvendor",
        "products": products,
        "global_facts": {},
        "objective": {
            "sense": "maximize",
            "component_ids": [
                "sales_revenue",
                "purchase_cost",
            ],
        },
        "constraints": [],
    }


def test_groups_missing_inputs_for_cards() -> None:
    product = create_product("product_1")

    del product["forecast"]["expected_value"]
    del product["facts"]["selling_price"]

    specification = create_specification([product])
    specification["constraints"] = [
        {
            "component_id": "minimum_order_quantity",
            "product_id": "product_1",
            "parameters": {},
        }
    ]

    layout = build_modular_input_layout(
        specification
    )

    assert layout["display_mode"] == "cards"
    assert layout["validation_errors"] == []

    assert len(layout["product_groups"]) == 1

    product_group = layout["product_groups"][0]

    assert product_group["product_id"] == "product_1"

    assert [
        missing_input["field"]["name"]
        for missing_input
        in product_group["forecast_inputs"]
    ] == [
        "expected_value",
    ]

    assert [
        missing_input["field"]["name"]
        for missing_input
        in product_group["fact_inputs"]
    ] == [
        "selling_price",
    ]

    assert len(layout["constraint_groups"]) == 1

    constraint_group = layout["constraint_groups"][0]

    assert (
        constraint_group["component_id"]
        == "minimum_order_quantity"
    )
    assert constraint_group["product_id"] == "product_1"

    assert (
        constraint_group["inputs"][0]["field"]["name"]
        == "quantity"
    )


def test_uses_table_for_many_products() -> None:
    products = [
        create_product(f"product_{index}")
        for index in range(30)
    ]

    for product in products:
        del product["facts"]["selling_price"]

    specification = create_specification(products)

    layout = build_modular_input_layout(
        specification
    )

    assert layout["display_mode"] == "table"
    assert len(layout["product_groups"]) == 30

    assert all(
        [
            missing_input["field"]["name"]
            for missing_input
            in product_group["fact_inputs"]
        ]
        == ["selling_price"]
        for product_group in layout["product_groups"]
    )


def test_switches_to_table_above_card_limit() -> None:
    three_products = create_specification(
        [
            create_product("product_1"),
            create_product("product_2"),
            create_product("product_3"),
        ]
    )

    four_products = create_specification(
        [
            create_product("product_1"),
            create_product("product_2"),
            create_product("product_3"),
            create_product("product_4"),
        ]
    )

    assert (
        build_modular_input_layout(
            three_products
        )["display_mode"]
        == "cards"
    )

    assert (
        build_modular_input_layout(
            four_products
        )["display_mode"]
        == "table"
    )


def test_applies_missing_values_and_completes_specification() -> None:
    product = create_product("product_1")
    del product["facts"]["selling_price"]

    specification = create_specification([product])

    initial_validation = validate_model_specification(
        specification
    )

    missing_input = initial_validation[
        "missing_inputs"
    ][0]

    updated_specification = apply_modular_input_value(
        specification,
        missing_input,
        12.0,
    )

    final_validation = validate_model_specification(
        updated_specification
    )

    assert final_validation["is_valid"] is True

    assert (
        updated_specification["products"][0]["facts"][
            "selling_price"
        ]
        == 12.0
    )

    assert (
        "selling_price"
        not in specification["products"][0]["facts"]
    )


def test_builds_input_for_selected_disposal_cost() -> None:
    product = create_product("product_1")
    specification = create_specification([product])

    specification["objective"]["component_ids"].append(
        "disposal_cost"
    )

    layout = build_modular_input_layout(specification)

    product_group = layout["product_groups"][0]
    fact_inputs = product_group["fact_inputs"]

    assert len(fact_inputs) == 1

    missing_input = fact_inputs[0]

    assert missing_input["component_id"] == "disposal_cost"
    assert missing_input["field"]["name"] == "disposal_cost"
    assert missing_input["field"]["label"] == (
        "Entsorgungskosten pro unverkaufter Einheit"
    )

    updated_specification = apply_modular_input_value(
        specification,
        missing_input,
        20.0,
    )

    assert (
        updated_specification["products"][0]["facts"][
            "disposal_cost"
        ]
        == 20.0
    )

    final_validation = validate_model_specification(
        updated_specification
    )

    assert final_validation["is_valid"] is True


def test_builds_input_for_missing_order_multiple() -> None:
    specification = create_specification(
        [
            create_product("product_1"),
        ]
    )

    specification["constraints"] = [
        {
            "component_id": "order_multiple",
            "product_id": "product_1",
            "parameters": {},
        }
    ]

    layout = build_modular_input_layout(specification)

    assert len(layout["constraint_groups"]) == 1

    constraint_group = layout["constraint_groups"][0]

    assert constraint_group["component_id"] == (
        "order_multiple"
    )
    assert constraint_group["product_id"] == "product_1"
    assert len(constraint_group["inputs"]) == 1

    missing_input = constraint_group["inputs"][0]

    assert missing_input["source"] == (
        "constraint_parameter"
    )
    assert missing_input["field"]["name"] == "multiple"
    assert missing_input["field"]["type"] == "integer"
    assert missing_input["field"]["unit"] == "Einheiten"

    updated_specification = apply_modular_input_value(
        specification,
        missing_input,
        60,
    )

    assert updated_specification["constraints"][0][
        "parameters"
    ]["multiple"] == 60

    final_validation = validate_model_specification(
        updated_specification
    )

    assert final_validation["is_valid"] is True


def test_builds_input_for_missing_cycle_service_level() -> None:
    specification = create_specification(
        [
            create_product("product_1"),
        ]
    )

    specification["constraints"] = [
        {
            "component_id": (
                "minimum_cycle_service_level"
            ),
            "product_id": "product_1",
            "parameters": {},
        }
    ]

    layout = build_modular_input_layout(
        specification
    )

    assert len(layout["constraint_groups"]) == 1

    constraint_group = layout["constraint_groups"][0]

    assert constraint_group["component_id"] == (
        "minimum_cycle_service_level"
    )
    assert constraint_group["product_id"] == "product_1"
    assert len(constraint_group["inputs"]) == 1

    missing_input = constraint_group["inputs"][0]

    assert missing_input["source"] == (
        "constraint_parameter"
    )
    assert missing_input["field"]["name"] == (
        "service_level_percent"
    )
    assert missing_input["field"]["type"] == "number"
    assert missing_input["field"]["unit"] == "%"
    assert missing_input["field"]["min_value"] == 0.01
    assert missing_input["field"]["max_value"] == 99.99

    updated_specification = apply_modular_input_value(
        specification,
        missing_input,
        85.0,
    )

    assert updated_specification["constraints"][0][
        "parameters"
    ]["service_level_percent"] == 85.0

    final_validation = validate_model_specification(
        updated_specification
    )

    assert final_validation["is_valid"] is True
