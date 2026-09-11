from agentic_optimization.models.specification import (
    ModelSpecification,
    ProductSpecification,
)
from agentic_optimization.ui.modular_table_builder import (
    build_product_table_rows,
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


def test_marks_missing_product_data_in_table_row() -> None:
    first_product = create_product("product_1")
    second_product = create_product("product_2")

    del second_product["forecast"]["quantiles"]
    del second_product["facts"]["selling_price"]

    specification = create_specification(
        [
            first_product,
            second_product,
        ]
    )

    rows = build_product_table_rows(specification)

    assert len(rows) == 2

    first_row = rows[0]
    second_row = rows[1]

    assert first_row["status"] == "complete"
    assert first_row["forecast_complete"] is True

    assert second_row["status"] == "incomplete"
    assert second_row["forecast_complete"] is False

    assert set(second_row["missing_field_names"]) == {
        "quantiles",
        "selling_price",
    }

    assert second_row["values"]["selling_price"] is None
    assert "quantiles" not in second_row["values"]


def test_builds_rows_for_thirty_products() -> None:
    specification = create_specification(
        [
            create_product(f"product_{index}")
            for index in range(30)
        ]
    )

    rows = build_product_table_rows(specification)

    assert len(rows) == 30

    assert [
        row["product_id"]
        for row in rows
    ] == [
        f"product_{index}"
        for index in range(30)
    ]

    assert all(
        row["status"] == "complete"
        for row in rows
    )