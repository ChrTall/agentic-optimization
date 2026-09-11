import pulp
import pytest

from agentic_optimization.models.compiler import (
    compile_model_specification,
)
from agentic_optimization.models.specification import (
    ModelSpecification,
    ProductSpecification,
)
from agentic_optimization.models.execution import (
    execute_model_specification,
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


def solve(
    specification: ModelSpecification,
):
    context = compile_model_specification(
        specification
    )

    context.model.solve(
        pulp.PULP_CBC_CMD(msg=False)
    )

    assert pulp.LpStatus[context.model.status] == "Optimal"

    return context


def test_compiles_and_solves_single_product_model() -> None:
    specification = create_specification(
        [
            create_product("product_1"),
        ]
    )

    context = solve(specification)

    order_quantity = context.get_quantity(
        "order_quantity",
        "product_1",
    ).value()

    assert order_quantity == pytest.approx(100.0)
    assert pulp.value(
        context.model.objective
    ) == pytest.approx(592.0)


def test_applies_minimum_order_quantity_builder() -> None:
    specification = create_specification(
        [
            create_product("product_1"),
        ]
    )

    specification["constraints"] = [
        {
            "component_id": "minimum_order_quantity",
            "product_id": "product_1",
            "parameters": {
                "quantity": 110.0,
            },
        }
    ]

    context = solve(specification)

    order_quantity = context.get_quantity(
        "order_quantity",
        "product_1",
    ).value()

    assert order_quantity == pytest.approx(110.0)


def test_compiles_same_model_for_multiple_products() -> None:
    specification = create_specification(
        [
            create_product("product_1"),
            create_product("product_2"),
        ]
    )

    context = solve(specification)

    first_order_quantity = context.get_quantity(
        "order_quantity",
        "product_1",
    ).value()

    second_order_quantity = context.get_quantity(
        "order_quantity",
        "product_2",
    ).value()

    assert first_order_quantity == pytest.approx(100.0)
    assert second_order_quantity == pytest.approx(100.0)


def test_additional_component_changes_objective() -> None:
    specification = create_specification(
        [
            create_product("product_1"),
        ]
    )

    specification["products"][0]["facts"][
        "salvage_value"
    ] = 4.0

    specification["objective"]["component_ids"].append(
        "salvage_revenue"
    )

    context = solve(specification)

    order_quantity = context.get_quantity(
        "order_quantity",
        "product_1",
    ).value()

    assert order_quantity == pytest.approx(140.0)


def test_global_budget_couples_multiple_products() -> None:
    specification = create_specification(
        [
            create_product("product_1"),
            create_product("product_2"),
        ]
    )

    specification["constraints"] = [
        {
            "component_id": "budget_limit",
            "parameters": {
                "budget": 800.0,
            },
        }
    ]

    context = solve(specification)

    total_order_quantity = sum(
        context.get_quantity(
            "order_quantity",
            product_id,
        ).value()
        for product_id in [
            "product_1",
            "product_2",
        ]
    )

    assert total_order_quantity == pytest.approx(160.0)

    assert (
        5.0 * total_order_quantity
        <= 800.0 + 1e-6
    )


def test_applies_maximum_order_quantity_builder() -> None:
    specification = create_specification(
        [
            create_product("product_1"),
        ]
    )

    specification["constraints"] = [
        {
            "component_id": "maximum_order_quantity",
            "product_id": "product_1",
            "parameters": {
                "quantity": 80.0,
            },
        }
    ]

    context = solve(specification)

    order_quantity = context.get_quantity(
        "order_quantity",
        "product_1",
    ).value()

    assert order_quantity == pytest.approx(80.0)


def test_shortage_cost_changes_order_quantity() -> None:
    specification = create_specification(
        [
            create_product("product_1"),
        ]
    )

    specification["products"][0]["facts"][
        "shortage_cost"
    ] = 5.0

    specification["objective"]["component_ids"].append(
        "shortage_cost"
    )

    context = solve(specification)

    order_quantity = context.get_quantity(
        "order_quantity",
        "product_1",
    ).value()

    assert order_quantity == pytest.approx(140.0)


def test_global_storage_capacity_couples_products() -> None:
    first_product = create_product("product_1")
    second_product = create_product("product_2")

    first_product["facts"]["storage_space_per_unit"] = 1.0
    second_product["facts"]["storage_space_per_unit"] = 2.0

    specification = create_specification(
        [
            first_product,
            second_product,
        ]
    )

    specification["constraints"] = [
        {
            "component_id": "storage_capacity",
            "parameters": {
                "capacity": 210.0,
            },
        }
    ]

    context = solve(specification)

    first_quantity = context.get_quantity(
        "order_quantity",
        "product_1",
    ).value()

    second_quantity = context.get_quantity(
        "order_quantity",
        "product_2",
    ).value()

    used_capacity = (
        first_quantity
        + 2.0 * second_quantity
    )

    assert used_capacity <= 210.0 + 1e-6
    assert used_capacity == pytest.approx(210.0)


def test_executes_model_with_common_result_contract() -> None:
    specification = create_specification(
        [
            create_product("product_1"),
        ]
    )

    result = execute_model_specification(
        specification
    )

    assert result["status"] == "optimal"
    assert result["solution_method"] == "linear_programming"
    assert result["objective_value"] == pytest.approx(592.0)
    assert result["decisions"] == [
        {
            "quantity_id": "order_quantity",
            "product_id": "product_1",
            "value": pytest.approx(100.0),
        }
    ]
    assert result["metrics"] == [
        {
            "metric_id": "sold_quantity",
            "product_id": "product_1",
            "aggregation": "expected_value",
            "value": pytest.approx(91.0),
        },
        {
            "metric_id": "leftover_quantity",
            "product_id": "product_1",
            "aggregation": "expected_value",
            "value": pytest.approx(9.0),
        },
        {
            "metric_id": "shortage_quantity",
            "product_id": "product_1",
            "aggregation": "expected_value",
            "value": pytest.approx(12.0),
        },
    ]
    assert result["errors"] == []
    assert result["technical_representation"] is not None


def test_reports_infeasible_linear_model() -> None:
    specification = create_specification(
        [
            create_product("product_1"),
        ]
    )
    specification["constraints"] = [
        {
            "component_id": "minimum_order_quantity",
            "product_id": "product_1",
            "parameters": {
                "quantity": 110.0,
            },
        },
        {
            "component_id": "budget_limit",
            "parameters": {
                "budget": 0.0,
            },
        },
    ]

    result = execute_model_specification(
        specification
    )

    assert result["status"] == "infeasible"
    assert result["solution_method"] == "linear_programming"
    assert result["objective_value"] is None
    assert result["decisions"] == []
    assert result["errors"]
