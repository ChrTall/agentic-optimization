import pulp
import pytest

from agentic_optimization.models.forecast_adapter import (
    DEMAND_SCENARIO_COUNT,
)

from agentic_optimization.models.newsvendor.single_period import (
    build_single_period_newsvendor_core,
)
from agentic_optimization.models.specification import (
    ModelSpecification,
    ProductSpecification,
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


def test_builds_core_for_single_product() -> None:
    specification = create_specification(
        [
            create_product("product_1"),
        ]
    )

    result = build_single_period_newsvendor_core(
        specification
    )

    assert result.model.sense == pulp.LpMaximize

    assert set(result.order_quantities) == {
        "product_1",
    }

    assert len(result.sold_quantities) == DEMAND_SCENARIO_COUNT
    assert len(result.leftover_quantities) == DEMAND_SCENARIO_COUNT
    assert len(result.shortage_quantities) == DEMAND_SCENARIO_COUNT

    assert len(result.model.constraints) == 2 * DEMAND_SCENARIO_COUNT

    order_quantity = result.get_quantity(
        "order_quantity",
        "product_1",
    )

    sold_quantity = result.get_quantity(
        "sold_quantity",
        "product_1",
        0,
    )

    assert order_quantity is result.order_quantities["product_1"]

    assert (
        sold_quantity
        is result.sold_quantities[("product_1", 0)]
    )


def test_builds_same_core_for_multiple_products() -> None:
    specification = create_specification(
        [
            create_product("product_1"),
            create_product("product_2"),
        ]
    )

    result = build_single_period_newsvendor_core(
        specification
    )

    assert set(result.order_quantities) == {
        "product_1",
        "product_2",
    }

    assert len(result.sold_quantities) == 2 * DEMAND_SCENARIO_COUNT
    assert len(result.leftover_quantities) == 2 * DEMAND_SCENARIO_COUNT
    assert len(result.shortage_quantities) == 2 * DEMAND_SCENARIO_COUNT

    assert len(result.model.constraints) == 4 * DEMAND_SCENARIO_COUNT


def test_uses_prepared_forecast_for_each_product() -> None:
    specification = create_specification(
        [
            create_product("product_1"),
            create_product("product_2"),
        ]
    )

    result = build_single_period_newsvendor_core(
        specification
    )

    first_forecast = result.forecasts["product_1"]

    assert len(first_forecast["demand_points"]) == DEMAND_SCENARIO_COUNT

    assert sum(
        point["probability"]
        for point in first_forecast["demand_points"]
    ) == pytest.approx(1.0)


def test_rejects_expected_demand_that_contradicts_quantiles() -> None:
    from agentic_optimization.models.newsvendor.single_period import (
        validate_single_period_newsvendor_forecast,
    )

    forecast = {
        "expected_value": 7300.0,
        "quantiles": [
            {"level": 0.1, "value": 70.0},
            {"level": 0.5, "value": 100.0},
            {"level": 0.9, "value": 140.0},
        ],
        "horizon_days": 30,
    }

    errors = validate_single_period_newsvendor_forecast(forecast)

    assert len(errors) == 1
    assert "denselben Prognosezeitraum" in errors[0]

    forecast["expected_value"] = 105.0

    assert validate_single_period_newsvendor_forecast(forecast) == []
