from copy import deepcopy

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
        "core_id": "continuous_review_q_r",
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
                    "lead_time_days": 10,
                    "ordering_cost": 50.0,
                    "holding_cost": 2.0,
                    "shortage_cost": 20.0,
                },
            }
        ],
        "global_facts": {},
        "objective": {
            "sense": "minimize",
            "component_ids": [
                "annual_ordering_cost",
                "annual_holding_cost",
                "annual_shortage_cost",
            ],
        },
        "constraints": [],
    }


def decision_values(
    result: dict,
) -> dict[str, float]:
    return {
        decision["quantity_id"]: decision["value"]
        for decision in result["decisions"]
    }


def test_executes_continuous_review_q_r() -> None:
    specification = create_specification()

    validation = validate_model_specification(
        specification
    )
    result = execute_model_specification(specification)

    assert validation["is_valid"] is True
    assert result["status"] == "optimal"
    assert result["solution_method"] == "numerical"
    assert result["objective_value"] == pytest.approx(
        569.3738014
    )
    assert decision_values(result) == {
        "order_quantity": 253,
        "reorder_point": 65,
    }
    assert result["errors"] == []


def test_exposes_q_r_result_metrics() -> None:
    result = execute_model_specification(
        create_specification()
    )

    metric_ids = {
        metric["metric_id"]
        for metric in result["metrics"]
    }

    assert {
        "annual_demand",
        "lead_time_demand_mean",
        "lead_time_demand_std",
        "orders_per_year",
        "safety_stock",
        "average_inventory",
        "expected_shortage_per_cycle",
        "annual_shortage_quantity",
        "annual_ordering_cost",
        "annual_holding_cost",
        "annual_shortage_cost",
        "annual_relevant_cost",
    } <= metric_ids

    assert "G(Q,R)" in result["technical_representation"]


def test_applies_q_r_constraints() -> None:
    specification = create_specification()
    specification["constraints"] = [
        {
            "component_id": "minimum_order_quantity",
            "product_id": "product_1",
            "parameters": {
                "quantity": 300.0,
            },
        },
        {
            "component_id": "maximum_order_quantity",
            "product_id": "product_1",
            "parameters": {
                "quantity": 320.0,
            },
        },
        {
            "component_id": "minimum_reorder_point",
            "product_id": "product_1",
            "parameters": {
                "reorder_point": 70.0,
            },
        },
        {
            "component_id": "maximum_reorder_point",
            "product_id": "product_1",
            "parameters": {
                "reorder_point": 80.0,
            },
        },
    ]

    result = execute_model_specification(specification)

    assert decision_values(result) == {
        "order_quantity": 300,
        "reorder_point": 70,
    }


def test_rejects_missing_q_r_cost_input() -> None:
    specification = create_specification()
    specification["products"][0]["facts"].pop(
        "shortage_cost"
    )

    validation = validate_model_specification(
        specification
    )

    assert validation["is_valid"] is False
    assert any(
        missing_input["field"]["name"]
        == "shortage_cost"
        for missing_input in validation["missing_inputs"]
    )

    with pytest.raises(
        ValueError,
        match="shortage_cost",
    ):
        execute_model_specification(specification)


def test_executes_multiple_products_independently() -> None:
    specification = create_specification()

    second_product = deepcopy(
        specification["products"][0]
    )
    second_product["product_id"] = "product_2"
    second_product["facts"]["ordering_cost"] = 80.0

    specification["products"].append(second_product)
    specification["constraints"] = [
        {
            "component_id": "minimum_order_quantity",
            "product_id": "product_2",
            "parameters": {
                "quantity": 400.0,
            },
        }
    ]

    result = execute_model_specification(specification)

    decisions = {
        (
            decision["product_id"],
            decision["quantity_id"],
        ): decision["value"]
        for decision in result["decisions"]
    }

    assert len(decisions) == 4

    assert decisions[
        ("product_1", "order_quantity")
    ] == 253
    assert decisions[
        ("product_1", "reorder_point")
    ] == 65

    assert decisions[
        ("product_2", "order_quantity")
    ] >= 400
    assert decisions[
        ("product_2", "reorder_point")
    ] >= 0

    product_costs = [
        metric["value"]
        for metric in result["metrics"]
        if metric["metric_id"] == "annual_relevant_cost"
    ]

    assert len(product_costs) == 2
    assert result["objective_value"] == pytest.approx(
        sum(product_costs)
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
    decisions = decision_values(result)

    assert validation["is_valid"] is True
    assert decisions["order_quantity"] == 300
    assert decisions["order_quantity"] % 100 == 0
    assert decisions["reorder_point"] == 64
    assert result["objective_value"] == pytest.approx(
        576.6824991
    )
